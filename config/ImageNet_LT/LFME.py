# Testing configurations
config = {}
import os
name = __file__.split('/')[-1].split('.')[-2]
training_opt = {}
training_opt['dataset'] = 'ImageNet_LT'
training_opt['distill_shot_phases'] = ['low_shot', 'median_shot', 'many_shot']
# training_opt['log_dir'] = './logs/ImageNet_LT/distill_sampler_weight'
training_opt['log_dir'] = os.path.join('./logs/ImageNet_LT/', name)
# training_opt['log_dir_low_shot'] = './logs/ImageNet_LT/low_shot/'
# training_opt['log_dir_median_shot'] = './logs/ImageNet_LT/median_shot/'
# training_opt['log_dir_many_shot'] = './logs/ImageNet_LT/many_shot/'

training_opt['num_classes'] = 1000
training_opt['num_classes_low_shot'] = 146
training_opt['num_classes_median_shot'] = 463
training_opt['num_classes_many_shot'] = 391

training_opt['batch_size'] = 256
training_opt['num_workers'] = 4
training_opt['num_epochs'] = 90
training_opt['display_step'] = 10
training_opt['feature_dim'] = 512
training_opt['open_threshold'] = 0.1
training_opt['instance_scheduler_full_epoch'] = 1
training_opt['decay_thresh'] = 0.3
# training_opt['sampler'] = None
# training_opt['sampler'] = {'type': 'ClassAwareSampler', 'def_file': './data/ClassAwareSampler.py',
#                            'num_samples_cls': 4}
training_opt['sampler'] = {'type': 'CurricSampler', 'def_file': './data/CurricSampler.py',
                           'num_samples_cls': 4}
training_opt['shot_phase'] = None
training_opt['scheduler_params'] = {'lr_scheduler_type':'MultiStepLR', 'milestones':[50, 80], 'gamma':0.1}
config['training_opt'] = training_opt

networks = {}
feature_param = {'use_modulatedatt':False, 'use_fc': True, 'dropout': None,
        'stage1_weights': False, 'dataset': training_opt['dataset']}
feature_optim_param = {'lr': 0.1, 'momentum':0.9, 'weight_decay':0.0005}
networks['feat_model'] = {'def_file': './models/ResNet10Feature.py',
                          'params': feature_param,
                          'optim_params': feature_optim_param,
                          'fix': False}


feature_param_low_shot = {'use_modulatedatt':False, 'use_fc': True, 'dropout': None,
        'stage1_weights': True, 'dataset': training_opt['dataset'], 'shot_phase': 'low_shot'}

feature_param_median_shot = {'use_modulatedatt':False, 'use_fc': True, 'dropout': None,
        'stage1_weights': True, 'dataset': training_opt['dataset'], 'shot_phase': 'median_shot'}

feature_param_many_shot = {'use_modulatedatt':False, 'use_fc': True, 'dropout': None,
        'stage1_weights': True, 'dataset': training_opt['dataset'], 'shot_phase': 'many_shot'}

networks['feat_model_low_shot'] = {'def_file': './models/ResNet10Feature.py',
                                   'params': feature_param_low_shot,
                                   'shot_phase':'low_shot'}
networks['feat_model_median_shot'] = {'def_file': './models/ResNet10Feature.py',
                                      'params': feature_param_median_shot,
                                      'shot_phase':'median_shot'}
networks['feat_model_many_shot'] = {'def_file': './models/ResNet10Feature.py',
                                    'params': feature_param_many_shot,
                                    'shot_phase':'many_shot'}


classifier_param = {'in_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes'],
        'stage1_weights': False, 'dataset': training_opt['dataset']}

classifier_param_low_shot = {'in_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes_low_shot'],
        'stage1_weights': True, 'dataset': training_opt['dataset'], 'shot_phase': 'low_shot'}

classifier_param_median_shot = {'in_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes_median_shot'],
        'stage1_weights': True, 'dataset': training_opt['dataset'], 'shot_phase': 'median_shot'}

classifier_param_many_shot = {'in_dim': training_opt['feature_dim'], 'num_classes': training_opt['num_classes_many_shot'],
        'stage1_weights': True, 'dataset': training_opt['dataset'], 'shot_phase': 'many_shot'}

classifier_optim_param = {'lr': 0.1, 'momentum':0.9, 'weight_decay':0.0005}
networks['classifier'] = {'def_file': './models/DotProductClassifier.py',
                          'params': classifier_param,
                          'optim_params': classifier_optim_param}
networks['classifier_low_shot'] = {'def_file': './models/DotProductClassifier.py',
                          'params': classifier_param_low_shot}
networks['classifier_median_shot'] = {'def_file': './models/DotProductClassifier.py',
                          'params': classifier_param_median_shot}
networks['classifier_many_shot'] = {'def_file': './models/DotProductClassifier.py',
                          'params': classifier_param_many_shot}

config['networks'] = networks

criterions = {}
perf_loss_param = {}
distill_loss_param = {'T': 2.0}
criterions['PerformanceLoss'] = {'def_file': './loss/SoftmaxLoss.py', 'loss_params': perf_loss_param,
                                 'optim_params': None, 'weight': 1.0}
criterions['DistillLoss'] = {'def_file': './loss/DistillationLoss.py', 'loss_params': distill_loss_param,
                                 'optim_params': None, 'weight': 1.0}
config['criterions'] = criterions

memory = {}
memory['centroids'] = False
memory['init_centroids'] = False
config['memory'] = memory