# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# https://github.com/michuanhaohao/reid-strong-baseline/blob/master/solver/lr_scheduler.py
from bisect import bisect_right
import torch
import math

# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class AntiCosLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(AntiCosLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr - (self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2)
                for base_lr in self.base_lrs]
