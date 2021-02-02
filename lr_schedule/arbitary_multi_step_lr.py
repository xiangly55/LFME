# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# https://github.com/michuanhaohao/reid-strong-baseline/blob/master/solver/lr_scheduler.py
from bisect import bisect_right
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class ArbitartyMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        if not len(list(gammas))-1 == len(list(milestones)):
            raise ValueError('Milestones should have same length as gammas - 1 ')
        
        self.milestones = milestones
        self.gammas = gammas
        super(ArbitartyMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gammas[bisect_right(self.milestones, self.last_epoch)]
                for base_lr in self.base_lrs]
