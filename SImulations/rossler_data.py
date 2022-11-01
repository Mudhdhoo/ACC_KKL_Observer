import sys
import torch
import Systems
import numpy as np
from Dataset import DataSet
import scipy.io as spio
import Observer
from Observer import System_z, Observer
from data_generation import listdir_filter

instructions = """To run the script, create a path and download the repository contents. Place the trained models
in a seperate directory. Run python duffing_data.py arg1 arg2
    arg1 = Location of model directory.
    arg2 = Location to store the generated data."""

def main():
    model_path = sys.argv[1]
    model_names = listdir_filter(model_path)
    save_dir = sys.argv[2]
    
    models = {}
    for model in model_names:
        loaded_model = torch.load(model_path+'/'+model)
        models[model] = loaded_model

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
    z_sys = System_z(A, B, rossler)
    dataset = DataSet(rossler, A, B, a, b, N, num_ic, limits, PINN_sample_mode = 'split traj', data_gen_mode = 'negative forward')

    # --------------------- Generate Data ---------------------

    print('Running simulations.', '\n')
    np.random.seed(888)

    ic = 50       # Number trajectories
    ic_samples_in = np.random.uniform(-1, 1, (ic,1,3))
    ic_samples_out = np.concatenate((np.random.uniform(1, 1.5, (int(ic/2),1,3)), np.random.uniform(-1, -1.5, (int(ic/2),1,3)))).reshape(ic, 3)
    np.random.shuffle(ic_samples_out[:,0])
    np.random.shuffle(ic_samples_out[:,1])
    np.random.shuffle(ic_samples_out[:,2])
    ic_samples_out = np.expand_dims(ic_samples_out, axis = 1)
    ic_train = np.expand_dims(dataset.ic, axis = 1)

    mdict_out = {}
    mdict_in = {}
    mdict_train = {}
    for model in models:
        observer = Observer(rossler, z_sys, models[model], a, b, N)

        x_in, x_hat_in, _, _, t = observer.sim_multi(ic_samples_in)
        x_out, x_hat_out, _, _, t = observer.sim_multi(ic_samples_out)
        x_train, x_hat_train, _, _, t = observer.sim_multi(ic_train)

        mdict_out['x_hat_{}_out_train_test'.format(model)] = x_hat_out
        mdict_in['x_hat_{}_in_train_test'.format(model)] = x_hat_in
        mdict_train['x_hat_{}_train'.format(model)] = x_hat_train

    mdict_in['x_in_train_test'] = x_in
    mdict_out['x_out_train_test'] = x_out
    mdict_train['x_train'] = x_train

    spio.savemat(save_dir+'/rossler_test_in.mat',mdict_in)
    spio.savemat(save_dir+'/rossler_test_out.mat',mdict_out)
    spio.savemat(save_dir+'/rossler_train.mat',mdict_train)
    print('Simulations complete, data saved to directory.')
    
if __name__ =='__main__':
    try:
        main()
    except Exception as e:
        print(e, '\n')
        print(instructions)

