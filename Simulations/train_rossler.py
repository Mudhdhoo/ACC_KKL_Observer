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
    a0 = 0.2
    b0 = 0.2
    c0 = 5.7
    limits = limits = np.array([[-1,1], [-1,1], [-1,1]])    # Sample space

    a = 0   # start
    b = 50  # end
    N = 1000        # Number of intervals for RK4
    num_ic = 50

    # A and B matricies 
    A = np.array([[-2.12116939,  2.73877907, -0.75338041,  3.13947511,  1.00581224,
            -5.34548426, -5.34544528],
        [-5.91811463, -1.99160105, -0.26953991,  0.45654273, -0.56750459,
            -3.79023002, -3.00480715],
        [-1.27814843, -2.08375366, -2.60781906, -4.11186517,  0.33087248,
            0.08191241, -2.81998688],
        [ 0.69080313,  2.00687664,  3.02478991, -3.53231915, -0.89704804,
            -5.19153027,  3.16796666],
        [-1.16566857, -0.53979687, -0.30300461, -3.79753229, -2.7939018 ,
            -0.30176078, -2.61480264],
        [ 2.97899741,  3.97265572, -2.04951338, -1.08716448, -0.98741222,
            -4.7047881 , -1.38218644],
        [ 0.95887295,  0.43056828, -1.39564903, -3.18608794,  3.74388666,
            -3.88509855, -3.07468653]])

    B = np.ones([7,1])
    rossler = Systems.Rossler(a0, b0, c0)
    dataset = DataSet(rossler, A, B, a, b, N, num_ic, limits, PINN_sample_mode = 'split traj', data_gen_mode = 'negative forward')

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
    lmbda = 0.1
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

