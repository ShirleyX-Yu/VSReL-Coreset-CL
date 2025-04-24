# -*-coding:utf8-*-

import torch


class CompliedLoss(torch.nn.Module):
    def __init__(self, ce_factor, mse_factor, reduction='mean', kd_mode='mse'):
        super(CompliedLoss, self).__init__()
        self.reduction = reduction
        self.ce_factor = ce_factor
        self.mse_factor = mse_factor
        self.kd_mode = kd_mode
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
        if self.kd_mode == 'mse':
            self.mse_loss = torch.nn.MSELoss(reduction=reduction)
        elif self.kd_mode == 'ce':
            self.mse_loss = KDCrossEntropyLoss(reduction=reduction)
        else:
            raise ValueError('not a valid model')

    def forward(self, x, y, logits=None):
        loss_c = self.ce_factor * self.ce_loss(x, y)
        if self.mse_factor > 0 and logits is not None:
            loss_m = self.mse_loss(x, logits)
            if self.reduction == 'none':
                loss_m = torch.mean(loss_m, dim=-1)
            loss = self.ce_factor * loss_c + self.mse_factor * loss_m
            return loss
        else:
            return self.ce_factor * loss_c


class KDCrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction):
        super(KDCrossEntropyLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        py = self.softmax(y)
        px = self.softmax(x)
        loss = torch.sum(py * torch.log(px), dim=-1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)
