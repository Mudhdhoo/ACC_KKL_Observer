import sys
import torch
from torch import nn
import Systems
import numpy as np
from NN import Main_Network
from Dataset import DataSet
from Normalizer import Normalizer
from Trainer import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

instructions = """To run the script, create a path and download the repository contents.
Run python train_duffing.py arg1 arg2
    arg1 = Location to store the trained model.
    arg2 = Training method, either supervised_NN, unsupervised_AE or supervised_PINN."""

def main():
    torch.manual_seed(9)

    methods = ['supervised_NN', 'unsupervised_AE', 'supervised_PINN']
    save_dir = sys.argv[1]
    method = sys.argv[2]
    if method not in methods:
        raise Exception('Invalid choice of method')

    # --------------------- System Setup --------------------- 

    print('Generating Data.', '\n')
    limits_normal = np.array([[-1, 1], [-1, 1]])    # Sample space for normal datapoints
    a = 0   # start
    b = 50  # end
    N = 1000          # Number of intervals for RK4
    num_ic = 50       # Number of initial conditions to be sampled

    # A and B matricies 
    A = np.array([-6.5549,  4.6082, -5.2057, 3.3942, 6.0211,
                  -10.9772, -2.3362, -3.7164, -3.9566, -3.7166,
                  -1.9393, -0.2797, -2.7983, -0.8606, -4.8050,
                  -10.5100, -1.0820, -2.6448, -2.1144, -7.0080,
                  -10.1003, -0.5111, 1.0275, 3.1996, -0.3463]).reshape(5,5)

    B = np.ones([5,1])

    revduff = Systems.RevDuff(5, add_noise=False)
    dataset = DataSet(revduff, A, B, a, b, N, num_ic, limits_normal, PINN_sample_mode = 'split traj', data_gen_mode = 'backward sim')
    print('Dataset sucessfully generated.', '\n')

    # --------------------- Training Setup ---------------------
    x_size = dataset.system.x_size
    z_size = dataset.system.z_size
    num_hidden = 5
    hidden_size = 50
    activation = F.relu
    normalizer = Normalizer(dataset)
    main_net = Main_Network(x_size, z_size, num_hidden, hidden_size, activation, normalizer)      

    epochs = 15
    learning_rate = 0.001
    batch_size = 32
    lmbda = 1
    optimizer = torch.optim.Adam(main_net.parameters(), lr = learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 1, threshold = 0.0001, verbose = True)
    loss_fn = nn.MSELoss(reduction = 'mean')

    trainer = Trainer(dataset, epochs, optimizer, main_net, loss_fn, batch_size, lmbda, method, scheduler = scheduler)
    trainer.train()       
    torch.save(main_net, save_dir+'/'+method)

    print('Training complete.', '\n')
    
if __name__ =='__main__':
    try:
        main()
    except Exception as e:
        print(e)
        print(instructions)


