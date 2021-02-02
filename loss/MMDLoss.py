

import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul, kernel_num, fix_sigma):
        super(MMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def guassian_kernel(self, source, target):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)  # /len(kernel_val)

    def DAN(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target)
        loss = 0
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            # print('kernels', kernels[s1, s2], kernels[t1, t2], kernels[s1, t2], kernels[s2, t1])
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
            # print('loss', loss)

        # print('loss **************', loss)
        return loss / float(batch_size)

    def DAN_Linear(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target)

        # Linear version
        loss = 0
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        return loss / float(batch_size)

    def forward(self, source, target):
        return self.DAN(source, target)


def create_loss(kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    print('Loading MMD Loss.')
    return MMDLoss(kernel_mul, kernel_num, fix_sigma)