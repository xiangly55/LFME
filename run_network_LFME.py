import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
import time
import numpy as np
import warnings
import pdb
from torch.utils.tensorboard import SummaryWriter
from lr_schedule.warm_up_lr import WarmupMultiStepLR

class model():

    def __init__(self, config, data, test=False):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = self.config['training_opt']
        self.memory = self.config['memory']
        self.data = data
        self.test_mode = test
        self.distill_shot_phases = self.training_opt['distill_shot_phases']
        self.shot_list = get_shot_list()

        self.writer = SummaryWriter(log_dir=self.training_opt['log_dir'])
        # Initialize model
        self.init_models()
        self.init_all_distill_models()
        torch.backends.cudnn.benchmark = False
        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:
            # If using steps for training, we need to calculate training steps
            # for each epoch based on actual number of training data instead of
            # oversampled data number
            print('Using steps for training.')
            self.training_data_num = len(self.data['train'].dataset)
            self.epoch_steps = int(self.training_data_num \
                                   / self.training_opt['batch_size'])

            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            self.scheduler_params = self.training_opt['scheduler_params']
            self.model_optimizer, \
            self.model_optimizer_scheduler = self.init_optimizers(self.model_optim_params_list)
            self.init_criterions()
            if self.memory['init_centroids']:
                self.criterions['FeatureLoss'].centroids.data = \
                    self.centroids_cal(self.data['train_plain'])

        # Set up log file
        # self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
        self.log_file = get_logfile_name(self.training_opt['log_dir'])
        print_write_config(str(config), self.log_file)
        # if os.path.isfile(self.log_file):
        #     os.remove(self.log_file)
        # self.training_opt['shot_phase'] = 'many_shot'
        # self.networks['feat_model'] = self.distill_networks['many_shot']['feat_model_many_shot']
        # self.networks['classifier'] = self.distill_networks['many_shot']['classifier_many_shot']
        # self.training_opt['num_classes'] = 391
        # self.eval()
        self.distill_acc = {}
        if not self.test_mode:
            for shot_phase in ['low_shot', 'median_shot', 'many_shot']:
                self.distill_acc[shot_phase] = self.eval_distill_model(shot_phase)
            self.eval_instance_hardness()
        self.current_acc_top1 = {'many_shot':0.0,
                                 'median_shot': 0.0,
                                 'low_shot': 0.0}
        print(self.distill_acc)


    def init_models(self, optimizer=True):

        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        print("Using", torch.cuda.device_count(), "GPUs.")
        print('networks_defs', networks_defs)

        for key, val in networks_defs.items():
            # Networks
            if 'shot' in key:
                continue
            def_file = val['def_file']
            model_args = list(val['params'].values())
            model_args.append(self.test_mode)

            print('model_args', model_args)
            self.networks[key] = source_import(def_file).create_model(*model_args)
            self.networks[key] = nn.DataParallel(self.networks[key]).to(self.device)

            if 'fix' in val and val['fix']:
                print('Freezing feature weights except for self attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'selfatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False

            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                 'lr': optim_params['lr'],
                                                 'momentum': optim_params['momentum'],
                                                 'weight_decay': optim_params['weight_decay']})

    def init_all_distill_models(self):
        self.distill_networks = {}
        for shot_phase in self.distill_shot_phases:
            self.distill_networks[shot_phase] = self.init_distill_models(shot_phase)

    def init_distill_models(self, shot_phase):

        networks_defs = self.config['networks']
        distill_network = {}
        # self.model_optim_params_list = []

        # print("Using", torch.cuda.device_count(), "GPUs.")
        # print('networks_defs', networks_defs)

        classifier_name = 'classifier_' + shot_phase
        feat_model_name = 'feat_model_' + shot_phase
        for key in [feat_model_name, classifier_name]:
            # Networks
            val = networks_defs[key]
            def_file = val['def_file']
            model_args = list(val['params'].values())
            model_args.append(self.test_mode)

            print('model_args', model_args)
            distill_network[key] = source_import(def_file).create_model(*model_args)
            distill_network[key] = nn.DataParallel(distill_network[key]).to(self.device)

            print('Freezing feature weights except for self attention weights (if exist).')
            for param_name, param in distill_network[key].named_parameters():
                # Freeze all parameters except self attention parameters
                # if 'selfatt' not in param_name and 'fc' not in param_name:
                    param.requires_grad = False

        return distill_network



    def init_criterions(self):

        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}
        self.criterion_weights_scheduler = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = val['loss_params'].values()
            self.criterions[key] = source_import(def_file).create_loss(*loss_args).to(self.device)
            self.criterion_weights[key] = val['weight']

            if 'scheduler' in val.keys():
                self.criterion_weights_scheduler[key] = val['scheduler']
            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                 'lr': optim_params['lr'],
                                 'momentum': optim_params['momentum'],
                                 'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)
        if self.scheduler_params['lr_scheduler_type'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.scheduler_params['step_size'],
                                              gamma=self.scheduler_params['gamma'])
        elif self.scheduler_params['lr_scheduler_type'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                  mode=self.scheduler_params['mode'],
                                                  factor=self.scheduler_params['factor'])
        elif self.scheduler_params['lr_scheduler_type'] == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=self.scheduler_params['milestones'],
                                                        gamma=self.scheduler_params['gamma'])
        elif self.scheduler_params['lr_scheduler_type'] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.training_opt['num_epochs'])

        elif self.scheduler_params['lr_scheduler_type'] == 'WarmupMultiStepLR':
            scheduler = WarmupMultiStepLR(optimizer, milestones=self.scheduler_params['milestones'], 
                                            gamma=self.scheduler_params['gamma'], warmup_iters=self.scheduler_params['warmup_iters'])

        return optimizer, scheduler




    def batch_forward(self, inputs, labels=None, centroids=False, feature_ext=False, phase='train'):
        '''
        This is a general single batch running function.
        '''

        # Calculate Features
        self.features, self.feature_maps = self.networks['feat_model'](inputs)


        self.features_distill = {}
        self.feature_maps_distill = {}
        for shot_phase in self.distill_shot_phases:
            self.features_distill[shot_phase], self.feature_maps_distill[shot_phase] = self.distill_networks[shot_phase]['feat_model_'+shot_phase](inputs)



        # If not just extracting features, calculate logits
        if not feature_ext:

            # During training, calculate centroids if needed to
            if phase != 'test':
                if centroids and 'FeatureLoss' in self.criterions.keys():
                    self.centroids = self.criterions['FeatureLoss'].centroids.data
                else:
                    self.centroids = None

            # Calculate logits with classifier
            self.logits, self.direct_memory_feature = self.networks['classifier'](self.features, self.centroids)

            self.logits_distill = {}
            for shot_phase in self.distill_shot_phases:
                self.logits_distill[shot_phase], _ = \
                    self.distill_networks[shot_phase]['classifier_' + shot_phase](self.features_distill[shot_phase], self.centroids)

    def batch_backward(self):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # Step optimizers
        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels):
        current_global_step = self.epoch * self.epoch_steps + self.step
        # First, apply performance loss
        self.logits = self.logits * self.scheduler_weight.unsqueeze(-1).cuda()
        self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels) \
                         * self.criterion_weights['PerformanceLoss']
        self.writer.add_scalar('loss/PerformanceLoss', self.loss_perf, current_global_step)

        self.loss_distill_tot = 0.0
        self.loss_distill = {}
        for shot_phase in self.distill_shot_phases:
            # mask = torch.ones(self.logits.size())
            self.loss_distill[shot_phase] = self.criterions['DistillLoss'](self.logits[:, self.shot_list[shot_phase]], \
                                                                    self.logits_distill[shot_phase], len(self.shot_list[shot_phase]))
            weight = self.get_distill_loss_weight(shot_phase)
            self.loss_distill_tot += self.loss_distill[shot_phase] * weight

            self.writer.add_scalar('loss/loss_raw_'+shot_phase, self.loss_distill[shot_phase], current_global_step)
            self.writer.add_scalar('loss/loss_coeff_' + shot_phase, weight, current_global_step)
            self.writer.add_scalar('loss/loss_with_coeff_' + shot_phase, self.loss_distill[shot_phase] * weight, current_global_step)

        # Add performance loss to total loss
        self.loss = self.loss_perf + self.loss_distill_tot * self.criterion_weights['DistillLoss']
        self.epoch_avg_losses += self.loss_perf

        # Apply loss on features if set up
        if 'FeatureLoss' in self.criterions.keys():
            self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
            # Add feature loss to total loss
            self.loss += self.loss_feat


    def get_distill_loss_weight(self, shot_phase):
        if self.current_acc_top1[shot_phase] < self.distill_acc[shot_phase] * self.training_opt['decay_thresh']:
            weight = 1.0
        else:
            gap = self.distill_acc[shot_phase] - self.current_acc_top1[shot_phase]
            weight = (gap / self.distill_acc[shot_phase]) / (1 - self.training_opt['decay_thresh'])
            weight = max(weight, 0.0)

        # if self.distill_loss_weights[shot_phase] < weight:
        #     weight = self.distill_loss_weights[shot_phase]
        return weight

    def instance_weight_scheduler(self, init_weight, scheduler_type='Linear'):
        '''
        E: total epochs, e: current epochs, w_i, initial weights
        Satisfies: e=0, weight=w_i, e=E, weight=1
        Linear: weight = (1 - w_i) / E * e
        Convex: weight = 1 - (1-w_i)*cos(e/E * pi/2)
        Concave: weight = (1-w_i)/log2 * (log(e/E+1))
        '''
        e = self.epoch
        E = self.total_epochs / 2
        if scheduler_type == 'Linear':
            weight = (1 - init_weight) / E * e + init_weight
        elif scheduler_type == 'Convex':
            weight = 1 - (1 - init_weight)*torch.cos(e / E * np.pi / 2)
        elif scheduler_type == 'Concave':
            weight = (1 - init_weight)/torch.log(2) * (torch.log1p(e / E))

        return weight


    def train(self):

        # When training the network
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        best_acc = 0.0
        best_epoch = 0

        end_epoch = self.training_opt['num_epochs']
        self.epoch_avg_losses = 0.0
        self.total_steps = self.training_opt['num_epochs'] * self.epoch_steps
        self.total_epochs = self.training_opt['num_epochs']
        
        self.distill_loss_weights = {'low_shot': 1.0, 'median_shot': 1.0, 'many_shot': 1.0}
        
        # Loop over epochs
        for epoch in range(1, end_epoch + 1):
            self.epoch = epoch
            for model in self.networks.values():
                model.train()

            torch.cuda.empty_cache()

            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            if 'lr_scheduler_type' not in self.scheduler_params.keys():
                self.scheduler_params['lr_scheduler_type'] = 'StepLR'

            if self.scheduler_params['lr_scheduler_type'] != 'ReduceLROnPlateau':
                self.model_optimizer_scheduler.step()
                if self.criterion_optimizer:
                    self.criterion_optimizer_scheduler.step()

            # Iterate over dataset
            for step, (inputs, labels, _, instance_weight) in enumerate(self.data['curric']['curric_loader']):
                self.step = step
                self.current_step = self.epoch_steps * self.epoch + self.step
                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):

                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(inputs, labels,
                                       centroids=self.memory['centroids'],
                                       phase='train')
                    self.scheduler_weight = self.instance_weight_scheduler(instance_weight)
                    self.batch_loss(labels)
                    self.batch_backward()

                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:
                        minibatch_loss_feat = self.loss_feat.item() \
                            if 'FeatureLoss' in self.criterions.keys() else None
                        minibatch_loss_perf = self.loss_perf.item()
                        _, preds = torch.max(self.logits, 1)
                        minibatch_acc = mic_acc_cal(preds, labels)

                        print_str = ['Epoch: [%d/%d]'
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d'
                                     % (step),
                                     'Minibatch_loss_feature: %.3f'
                                     % (minibatch_loss_feat) if minibatch_loss_feat else '',
                                     'Minibatch_loss_performance: %.3f'
                                     % (minibatch_loss_perf),
                                     'Minibatch_accuracy_micro: %.3f'
                                     % (minibatch_acc)]
                        print_write(print_str, self.log_file)


            # After every epoch, validation
            self.eval(phase='val')

            if self.scheduler_params['lr_scheduler_type'] == 'ReduceLROnPlateau':
                # self.model_optimizer_scheduler.step(self.eval_acc_mic_top1)
                self.epoch_avg_losses = self.epoch_avg_losses / self.epoch_steps
                self.model_optimizer_scheduler.step(self.epoch_avg_losses)
                self.epoch_avg_losses = 0.0

            # self.writer.add_scalar('loss_weight/DistillLoss', self.criterion_weights['DistillLoss'], self.epoch)

            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch = epoch
                best_acc = self.eval_acc_mic_top1
                best_centroids = self.centroids
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        print()
        print('Training Complete.')

        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        self.save_model(epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids)

        print('Done')

    def eval(self, phase='val', openset=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        if openset:
            print('Under openset test mode. Open threshold is %.1f'
                  % self.training_opt['open_threshold'])

        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_paths = np.empty(0)

        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):
                # In validation or testing
                self.batch_forward(inputs, labels,
                                   centroids=self.memory['centroids'],
                                   phase=phase)
                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))
                self.total_paths = np.concatenate((self.total_paths, paths))

        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        if openset:
            preds[probs < self.training_opt['open_threshold']] = -1
            self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
                                           self.total_labels[self.total_labels == -1])
            print('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1 = mic_acc_cal(preds[self.total_labels != -1],
                                             self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, openset=openset,
                                        theta=self.training_opt['open_threshold'])
        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1 = shot_acc(preds[self.total_labels != -1],
                                     self.total_labels[self.total_labels != -1],
                                     self.data['train'])

        self.current_acc_top1 = {'many_shot':self.many_acc_top1,
                                 'median_shot': self.median_acc_top1,
                                 'low_shot': self.low_acc_top1}

        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s'
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.3f'
                     % (self.eval_acc_mic_top1),
                     '\n',
                     'Averaged F-measure: %.3f'
                     % (self.eval_f_measure),
                     '\n',
                     'Many_shot_accuracy_top1: %.3f'
                     % (self.many_acc_top1),
                     'Median_shot_accuracy_top1: %.3f'
                     % (self.median_acc_top1),
                     'Low_shot_accuracy_top1: %.3f'
                     % (self.low_acc_top1),
                     '\n']

        if not self.test_mode:
            self.writer.add_scalar('acc/many_acc_top1', self.many_acc_top1, self.epoch)
            self.writer.add_scalar('acc/median_acc_top1', self.median_acc_top1, self.epoch)
            self.writer.add_scalar('acc/low_acc_top1', self.low_acc_top1, self.epoch)

        if phase == 'val':
            print_write(print_str, self.log_file)
        else:
            print(*print_str)

    def centroids_cal(self, data):

        centroids = torch.zeros(self.training_opt['num_classes'],
                                self.training_opt['feature_dim']).cuda()

        print('Calculating centroids.')

        for model in self.networks.values():
            model.eval()

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):

            for inputs, labels, _ in tqdm(data):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]

        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centroids

    def load_model(self):

        model_dir = os.path.join(self.training_opt['log_dir'],
                                 'final_model_checkpoint.pth')

        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))

        checkpoint = torch.load(model_dir)
        model_state = checkpoint['state_dict_best']

        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None

        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            # model.load_state_dict(model_state[key])
            model.load_state_dict(weights)


    def batch_forward_distill_model(self, inputs, labels, shot_phase, centroids=False, feature_ext=False, phase='val'):
        '''
        This is a general single batch running function.
        '''

        # Calculate Features
        self.features, self.feature_maps = self.distill_networks[shot_phase]['feat_model_'+shot_phase](inputs)
        # If not just extracting features, calculate logits

        centroids = None

        # Calculate logits with classifier
        # self.logits, self.direct_memory_feature = self.distill_networks[shot_phase]['classifier'](self.features, centroids)
        self.logits, _ = self.distill_networks[shot_phase]['classifier_'+shot_phase](self.features, centroids)

    def eval_distill_model(self, shot_phase, openset=False):
        phase = 'val'
        print_str = ['Phase: %s, Shot phase: %s' % (phase, shot_phase)]
        print(print_str)
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.distill_networks[shot_phase].values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes_'+shot_phase])).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_paths = np.empty(0)

        # Iterate over dataset
        # for inputs, labels, paths in tqdm(self.data[shot_phase]):
        for inputs, labels, paths in self.data[shot_phase]:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):
                # In validation or testing
                self.batch_forward_distill_model(inputs, labels, shot_phase,
                                   centroids=self.memory['centroids'],
                                   phase=phase)
                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))
                self.total_paths = np.concatenate((self.total_paths, paths))

        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        if openset:
            preds[probs < self.training_opt['open_threshold']] = -1
            self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
                                           self.total_labels[self.total_labels == -1])
            print('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        eval_acc_mic_top1 = mic_acc_cal(preds[self.total_labels != -1],
                                             self.total_labels[self.total_labels != -1])
        try:
            eval_f_measure = F_measure(preds, self.total_labels, openset=openset,
                                            theta=self.training_opt['open_threshold'])
        except:
            eval_f_measure = 0.0
        # self.many_acc_top1, \
        # self.median_acc_top1, \
        # self.low_acc_top1 = shot_acc(preds[self.total_labels != -1],
        #                              self.total_labels[self.total_labels != -1],
        #                              self.data['train'])


        # Top-1 accuracy and additional string

        print('Evaluation_accuracy_micro_top1: %.3f'
                     % (eval_acc_mic_top1))
        return eval_acc_mic_top1


    def concat_distill_outputs(self, distill_model_output, labels):
        # distill_model_output: dict
        num_classes = self.training_opt['num_classes']
        num_instances = distill_model_output['many_shot'].shape[0]
        # assert num_instances == 1
        result = np.zeros((num_instances, num_classes))

        for shot_phase in self.shot_list:
            result[:, self.shot_list[shot_phase]] = distill_model_output[shot_phase]
        labels_np = labels.cpu().numpy()
        return [result[i, labels_np[i]] for i in range(num_instances)]


    def eval_instance_hardness(self):
        # In validation or testing mode, set model to eval() and initialize running loss/correct
        tot_weights_path = 'tot_weights_' + self.training_opt['dataset'] + '.npy'
        if os.path.exists(tot_weights_path):
            tot_weights = np.load(tot_weights_path)
        else:
            for shot_phase in self.shot_list:
                for model in self.distill_networks[shot_phase].values():
                    model.eval()
            # self.total_logits = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
            # self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
            # self.total_paths = np.empty(0)

            tot_weights = np.zeros(len(self.data['curric']['eval_weight_loader'].dataset))
            # Iterate over dataset
            # for inputs, labels, paths in tqdm(self.data[shot_phase]):
            start = time.time()
            batch_size = self.training_opt['batch_size']
            for i, (inputs, labels, paths, _) in enumerate(self.data['curric']['eval_weight_loader']):
                # batch_size = inputs.size(0)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                distill_model_output = {}
                for shot_phase in self.shot_list:
                # If on training phase, enable gradients
                    with torch.set_grad_enabled(False):
                        # In validation or testing
                        self.batch_forward_distill_model(inputs, labels, shot_phase,
                                           centroids=self.memory['centroids'],
                                           phase='train')
                        distill_model_output[shot_phase] = F.softmax(self.logits, dim=1).cpu().numpy()
                if inputs.size(0) != batch_size:
                    assert i == len(self.data['curric']['eval_weight_loader']) - 1
                    tot_weights[i * batch_size:] = self.concat_distill_outputs(distill_model_output,
                                                                                                   labels)
                else:
                    tot_weights[i * batch_size:(i + 1) * batch_size] = self.concat_distill_outputs(distill_model_output,
                                                                                                   labels)
                        # self.total_logits = torch.cat((self.total_logits, self.logits))
                        # self.total_labels = torch.cat((self.total_labels, labels))
                        # self.total_paths = np.concatenate((self.total_paths, paths))
            end = time.time()
            np.save('tot_weights_' + self.training_opt['dataset'] + '.npy', tot_weights)
            print('Time used: %.3f s' % (end - start))

        self.data['curric']['curric_loader'].dataset.set_instance_weight(tot_weights)
        # self.data['curric']['curric_loader'].dataset.set_scale_classwise_instance_weight()
        self.data['curric']['curric_loader'].dataset.set_scale_classwise_instance_weight_with_shot(self.shot_list)

        # probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)
    # def load_all_distill_model(self, shot_phases):
    #     # Not used, load pretrained model, in creat_model() by param: use_stage1_weight=True
    #     ''' assume phases is a list containing all phases
    #         e.g. ['low_shot', 'median_shot', 'many_shot']
    #     '''
    #
    #     for shot_phase in shot_phases:
    #         model_dir_name = self.training_opt['log_dir_' + shot_phase]
    #         model_dir = os.path.join(model_dir_name, 'final_model_checkpoint.pth')
    #         self.load_distill_model(model_dir, shot_phase)
    #
    # def load_distill_model(self, model_dir, phase='low_shot'):
    #     print('Loading distill model from %s' % (model_dir))
    #
    #     checkpoint = torch.load(model_dir)
    #     model_state = checkpoint['state_dict_best']
    #
    #     # self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
    #
    #     for key, model in self.networks.items():
    #         weights = model_state[key]
    #         weights = {k: weights[k] for k in weights if k in model.state_dict()}
    #         # model.load_state_dict(model_state[key])
    #         model.load_state_dict(weights)
    #


    def save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):

        model_states = {'epoch': epoch,
                        'best_epoch': best_epoch,
                        'state_dict_best': best_model_weights,
                        'best_acc': best_acc,
                        'centroids': centroids}

        model_dir = os.path.join(self.training_opt['log_dir'],
                                 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)

    def output_logits(self, openset=False):
        filename = os.path.join(self.training_opt['log_dir'],
                                'logits_%s' % ('open' if openset else 'close'))
        print("Saving total logits to: %s.npz" % filename)
        np.savez(filename,
                 logits=self.total_logits.detach().cpu().numpy(),
                 labels=self.total_labels.detach().cpu().numpy(),
                 paths=self.total_paths)




# HOW TO VERIFY
# 1. Modify distill.py
# training_opt['shot_phase'] = 'many_shot'
# 2. Run following scripts:
# self.networks['feat_model'] = self.distill_networks['many_shot']['feat_model_many_shot']
# self.networks['classifier'] = self.distill_networks['many_shot']['classifier_many_shot']
# self.training_opt['num_classes'] = 391
# self.eval()
#