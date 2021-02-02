from models.ResNetFeature import *
from utils import *
        
def create_model(use_selfatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, caffe=False, shot_phase='stage1', test=False):
    
    print('Loading Scratch ResNet 152 Feature Model.')
    resnet152 = ResNet(Bottleneck, [3, 8, 36, 3], use_modulatedatt=use_selfatt, use_fc=use_fc, dropout=None)
    
    if not test:
        assert(caffe != stage1_weights)

        if caffe:
            print('Loading Caffe Pretrained ResNet 152 Weights.')
            resnet152 = init_weights(model=resnet152,
                                     weights_path='./logs/caffe_resnet152.pth',
                                     caffe=True)
        elif stage1_weights:
            assert(dataset)
            print('Loading %s %s ResNet 152 Weights.' % (dataset, shot_phase))
            resnet152 = init_weights(model=resnet152,
                                    weights_path='./logs/%s/%s/final_model_checkpoint.pth' % (dataset, shot_phase))
                                     # weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset)
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet152
