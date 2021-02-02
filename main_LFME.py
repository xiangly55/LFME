import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import argparse
import pprint
from data import dataloader_LFME as dataloader
from run_network_LFME import model
import warnings
from utils import source_import

# ================
# LOAD CONFIGURATIONS

# data_root = {'ImageNet': '/home/public/public_dataset/ILSVRC2014/Img',
data_root = {'ImageNet': '/home/xiangliuyu/data/ILSVRC2015/Data/CLS-LOC/',
             'Places': '/home/zhmiao/datasets/Places365'}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/Imagenet_LT/Stage_1.py', type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
args = parser.parse_args()

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits

config = source_import(args.config).config
training_opt = config['training_opt']
if 'shot_phase' not in training_opt.keys():
    training_opt['shot_phase'] = None
# change
relatin_opt = config['memory']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
pprint.pprint(config)

if not test_mode:

    sampler_defs = training_opt['sampler']
    if sampler_defs:
        sampler_dic = {'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                       'num_samples_cls': sampler_defs['num_samples_cls']}
    else:
        sampler_dic = None

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=sampler_dic,
                                    num_workers=training_opt['num_workers'],
                                    shot_phase=training_opt['shot_phase'])
            for x in (['train', 'val', 'train_plain'] if relatin_opt['init_centroids'] else ['train', 'val'])}

    # get shot only data, for validation accuracy on distilled model
    for shot_phase in training_opt['distill_shot_phases']:
        data[shot_phase] = dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                             dataset=dataset, phase='val',
                                             batch_size=training_opt['batch_size'],
                                             sampler_dic=None,
                                             test_open=test_open,
                                             num_workers=training_opt['num_workers'],
                                             shuffle=False,
                                             shot_phase=shot_phase)
        data['curric'] = dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                             dataset=dataset, phase='train',
                                             batch_size=training_opt['batch_size'],
                                             sampler_dic=sampler_dic,
                                             test_open=test_open,
                                             num_workers=training_opt['num_workers'],
                                             shuffle=False,
                                             shot_phase=None,
                                             curric=True)
    training_model = model(config, data, test=False)

    training_model.train()

else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    print('Under testing phase, we load training data simply to calculate training data number for each class.')

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None,
                                    test_open=test_open,
                                    num_workers=training_opt['num_workers'],
                                    shuffle=False,
                                    shot_phase=training_opt['shot_phase'])
            for x in ['train', 'test']}

    training_model = model(config, data, test=True)
    training_model.load_model()
    training_model.eval(phase='test', openset=test_open)

    if output_logits:
        training_model.output_logits(openset=test_open)

print('ALL COMPLETED.')
