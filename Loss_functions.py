import torch
from torch import nn

class PdeLoss_xz(nn.Module):
    def __init__(self, M, K, system, loss_calculator, lmbda, reduction = 'mean'):
        super(PdeLoss_xz, self).__init__()
        self.M = M
        self.K = K
        self.system = system
        self.loss_calc = loss_calculator
        self.reduction = reduction
        self.lmbda = lmbda

    def forward(self, x, y, z_hat):
        loss = self.loss_calc.calc_pde_loss_xz(x, y, z_hat, self.system, self.M, self.K, self.reduction)
        return self.lmbda*loss

class PdeLoss_zx(nn.Module):
    def __init__(self, loss_calculator, reduction = 'mean'):
        super(PdeLoss_zx, self).__init__()
        self.loss_calc = loss_calculator
        self.reduction = reduction
        self.lmbda = 1

    def forward(self, x, z_hat):
        loss = self.loss_calc.calc_pde_loss_zx(x, z_hat, self.reduction)
        return self.lmbda*loss

class MSELoss(nn.Module):
    def __init__(self, loss_calculator):
        super(MSELoss, self).__init__()
        self.loss_calc = loss_calculator
        self.lmbda = 1

    def forward(self, x_hat, z_hat, x, z):
        loss = self.loss_calc.calc_loss(x_hat, z_hat, x, z)
        return self.lmbda*loss