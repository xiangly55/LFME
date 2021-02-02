import torch.nn as nn

def create_loss ():
    print('Loading NLLLoss.')
    return nn.NLLLoss()

