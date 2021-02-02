import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, T):
        super(DistillationLoss, self).__init__()
        self.T = T

    def forward(self, y, teacher_scores, scale):
        return F.kl_div(F.log_softmax(y / self.T), F.softmax(teacher_scores / self.T)) * scale

def create_loss(T=2.0):
    print('Loading Distillation Loss.')
    return DistillationLoss(T)



def distillation_loss(y, teacher_scores, T, scale):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """
    return F.kl_div(F.log_softmax(y / T), F.softmax(teacher_scores / T)) * scale
