import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from utils import *

class CosNorm_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, scale=16, *args):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = feat_dim
        self.out_dims = num_classes
        self.scale = scale
        self.weight = Parameter(torch.Tensor(self.out_dims, self.in_dims).cuda())
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input, 2, 1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t()), None
    
def create_model(feat_dim, num_classes=1000, scale=16, stage1_weights=False, dataset=None, shot_phase='stage1', test=False, *args):
    print('Loading Dot Product Classifier.')
    print(num_classes, feat_dim)
    clf = CosNorm_Classifier(num_classes, feat_dim, scale)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            clf.weight = init_weights(model=clf.weight,
                                  weights_path='./logs/%s/%s/final_model_checkpoint.pth' % (dataset, shot_phase),
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf