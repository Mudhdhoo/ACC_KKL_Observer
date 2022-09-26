import torch
import Loss_Calculator as L
from Loss_functions import *
from typing import TYPE_CHECKING, Optional
from Dataset import DataSet
from NN import Main_Network

class Trainer:
    """
    Trainer class containing the training loop.

    ---------------- Parameters ----------------
    dataset: Dataset
        Dataset object from Dataset.py.

    epochs: int
        Number of training epochs.

    optimizer: torch.optim
        Torch optimizer object.

    net: Main_Network
        Neural network object from NN.py.

    loss_fn: torch.nn
        Loss function from the torch.nn module.

    batch_size: int
        Dataloader batch size.

    shuffle: bool
        If True, the data is shuffled, otherwise not.

    scheduler: Optional
        Learning rate scheduler from torch.optim.lr_scheduler.

    reduction: str
        Either mean or sum.
        If mean, the mean loss of each batch is taken.
        If sum, the sum of the loss of each batch is used instead.
    """
    def __init__(self, dataset: DataSet, epochs: int, optimizer: torch.optim, net: Main_Network, loss_fn: torch.nn, batch_size: int, shuffle: bool = True, scheduler: Optional[None] = None, reduction: str = 'mean') -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.trainset = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net.to(self.device)
        self.loss_calculator = L.Loss_Calculator(loss_fn, self.net, self.dataset, self.device)
        self.normalizer = net.normalizer
        self.reduction = reduction
        self.pde1 = PdeLoss_xz(self.dataset.M, self.dataset.K, self.dataset.system, self.loss_calculator, self.reduction)
        #self.optimizer = torch.optim.Adam([{'params':self.net.parameters()}, {'params':self.pde1.lagrange}], lr = 0.001)
       # self.pde2 = PdeLoss_zx(self.loss_calculator, self.reduction)
        #self.optim2 = torch.optim.Adam(self.pde1.parameters(), lr = 0.0001)
        print('Device:', self.device)
        
    def train(self, with_pde: bool = True) -> None:
        """
        Training loop.
        """
        MSE = MSELoss(self.loss_calculator)
        for epoch in range(self.epochs):
            loss_sum = 0
            for idx, data in enumerate(self.trainset):
                x, z, y, x_ph, y_ph = data      # Normal and physics data
                x, z, y = x.to(self.device), z.to(self.device), y.to(self.device)
                if with_pde:
                    x_ph, y_ph = x_ph.to(self.device), y_ph.to(self.device)
                self.optimizer.zero_grad()
                self.net.mode = 'normal'
                z_hat, x_hat, norm_z_hat, norm_x_hat = self.net(x)
                if self.normalizer != None:
                    label_x = self.normalizer.Normalize(x, mode = 'normal').float()
                    label_z = self.normalizer.Normalize(z, mode = 'normal').float()
                else:
                    label_x = x
                    label_z = z

                # Compute MSE loss    
                loss_normal = MSE(norm_x_hat, norm_z_hat, label_x, label_z)

                # Compute physics loss
                if with_pde:
                    self.net.mode = 'physics'
                    z_hat_ph = self.net(x_ph)[0]
                    loss_pde1 = self.pde1(x_ph, y_ph, z_hat_ph)
                    #loss_pde2 = self.pde2(x_ph, z_hat_ph)
                    loss = loss_normal + loss_pde1 #+ loss_pde2
                else:
                    loss = loss_normal
    
                loss_sum += loss
                loss.backward()
                self.optimizer.step()
                #print(self.pde1.lagrange)
            training_loss = (loss_sum / idx).item()
            
            if self.scheduler == None:
                pass
            else:
                self.scheduler.step(training_loss)
            
            print('Epoch:', epoch+1, 'Loss:', training_loss)    # Average loss per epoch
                