import torch
from torch.autograd.functional import jacobian
import numpy as np

class Loss_Calculator:
    def __init__(self, loss_fn, net, dataset, device, method):
        self.loss_fn = loss_fn
        self.net = net
        self.device = device
        self.dataset = dataset
        self.method = method
        
    # Normal loss calculation
    def calc_loss(self, x_hat, z_hat, x, z):
        loss_xz = self.loss_fn(z_hat, z)
        loss_zx = self.loss_fn(x_hat, x)

        if self.method == 'unsupervised_AE':
            loss = loss_zx
        else:
            loss = loss_xz + loss_zx

        return loss
    
    # # PDE constrain loss for PINN from x --> z
    def calc_pde_loss_xz(self, x, y, z_hat, system, M, K, reduction = 'mean'):
        M = torch.from_numpy(M).to(self.device)
        K = torch.from_numpy(K).to(self.device)
        
        # Jacobian
        dTdx = self.calc_J(x, 'net1')
        
        # Computation of f(x)
        f = []
        u = 0
        for state in x:
            f.append(system.function(u, state.detach().numpy()))
        f = torch.from_numpy(np.array(f)).float().to(self.device)
        # dT/dx * f(x)
        dTdx_mul_f = torch.bmm(dTdx, torch.unsqueeze(f,2))

        z_hat = torch.unsqueeze(z_hat, 2)
        M = M.to(torch.float32)
        M_mul_T = torch.matmul(M, z_hat)    # MT(x)
        
        # Check if y elements are scalar
        K = K.to(torch.float32)
        y = y.to(torch.float32)
        if y[0].shape == torch.Size([]):
            K_mul_h = torch.matmul(K, y.view(y.shape[0],1,1))    # Kh(x)
        else:
            y = torch.unsqueeze(y, 2)
            K_mul_h = torch.matmul(K, y)    # Kh(x)
            
        pde = dTdx_mul_f - M_mul_T - K_mul_h    # dT/dx*f(x) - MT(x) - Kh(x) = 0
        loss_batch = torch.linalg.norm(pde, dim = 1)    # Element-wise norm

        # Type of loss reduction
        if reduction == 'mean':
            samples = loss_batch.shape[0]
            loss_pde = torch.sum(loss_batch) / samples
            
        if reduction == 'sum':
            loss_pde = torch.sum(loss_batch)
        
        return loss_pde
    
    # PDE constrain loss for PINN from z --> x    
    def calc_pde_loss_zx(self, x, z_hat, reduction = 'mean'):
        # Jacobian output of NN1 w.r.t input of NN1
        dTdx = self.calc_J(x, 'net1')

        # Jacobian output of NN2 w.r.t input of NN2
        dTheta_dT = self.calc_J(z_hat, 'net2')

        dTheta_dT_mul_dTdx = torch.bmm(dTheta_dT, dTdx) 
        
        pde = dTheta_dT_mul_dTdx - torch.eye(dTheta_dT_mul_dTdx.shape[1], dTheta_dT_mul_dTdx.shape[2]).to(self.device)    # dTheta/dT * dT/dx - I = 0
        
        loss_batch = torch.linalg.matrix_norm(pde)    # Element-wise matrix norm        

        # Type of loss reduction
        if reduction == 'mean':
            samples = loss_batch.shape[0]
            loss_pde = torch.sum(loss_batch) / samples
            
        if reduction == 'sum':
            loss_pde = torch.sum(loss_batch)
        
        return loss_pde

    # Jacobian calculation
    def calc_J(self, x, NN):
        m = x.shape[0]
        if NN == 'net1':
            net = self.net.net1
        if NN == 'net2':
             net = self.net.net2           
        dTdx = jacobian(net, x, create_graph=False)    # dT/dx   
        # result is m* d_o * m * d_i
        ind = torch.arange(0, m)
        
        return dTdx[ind, :, ind, :]
        