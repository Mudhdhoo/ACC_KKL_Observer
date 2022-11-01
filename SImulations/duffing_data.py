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
    np.random.seed(888)

    model_path = sys.argv[1]
    model_names = listdir_filter(model_path)
    save_dir = sys.argv[2]
    
    models = {}
    for model in model_names:
        loaded_model = torch.load(model_path+'/'+model)
        models[model] = loaded_model

    # --------------------- System Setup --------------------- 
    limits_normal = np.array([[-1, 1], [-1, 1]])    # Sample space for normal datapoints
    a = 0   # starting time
    b = 50  # end time
    N = 1000    # RK4 intervals
    num_ic = 50     # Number of initial conditions to be sampled

    # A and B matricies 
    A = np.array([-6.5549,  4.6082, -5.2057, 3.3942, 6.0211,
                  -10.9772, -2.3362, -3.7164, -3.9566, -3.7166,
                  -1.9393, -0.2797, -2.7983, -0.8606, -4.8050,
                  -10.5100, -1.0820, -2.6448, -2.1144, -7.0080,
                  -10.1003, -0.5111, 1.0275, 3.1996, -0.3463]).reshape(5,5)

    B = np.ones([5,1])

    revduff = Systems.RevDuff(5, add_noise=False)
    z_sys = System_z(A, B, revduff)
    dataset = DataSet(revduff, A, B, a, b, N, num_ic, limits_normal, PINN_sample_mode = 'split traj', data_gen_mode = 'backward sim')

   # --------------------- Generate Data ---------------------
    print('Running simulations.', '\n')
    num_ic = 50       
    ic_samples_in = np.random.uniform(-1, 1, (num_ic,1,2))
    ic_samples_out = np.concatenate((np.random.uniform(1.5, 3, (int(num_ic/2),1,2)), np.random.uniform(-1.5, -3, (int(num_ic/2),1,2)))).reshape(num_ic, 2)
    np.random.shuffle(ic_samples_out[:,0])
    np.random.shuffle(ic_samples_out[:,1])
    ic_samples_out = np.expand_dims(ic_samples_out, axis = 1)
    ic_train = np.expand_dims(dataset.ic, axis = 1)

    mdict_out = {}
    mdict_in = {}
    mdict_train = {}
    for model in models:
        observer = Observer(revduff, z_sys, models[model], a, b, N)

        x_in, x_hat_in, _, _, t = observer.sim_multi(ic_samples_in)
        x_out, x_hat_out, _, _, t = observer.sim_multi(ic_samples_out)
        x_train, x_hat_train, _, _, t = observer.sim_multi(ic_train)

        mdict_out['x_hat_{}_out_train_test'.format(model)] = x_hat_out
        mdict_in['x_hat_{}_in_train_test'.format(model)] = x_hat_in
        mdict_train['x_hat_{}_train'.format(model)] = x_hat_train

    mdict_in['x_in_train_test'] = x_in
    mdict_out['x_out_train_test'] = x_out
    mdict_train['x_train'] = x_train

    spio.savemat(save_dir+'/duffing_test_in.mat',mdict_in)
    spio.savemat(save_dir+'/duffing_test_out.mat',mdict_out)
    spio.savemat(save_dir+'/duffing_train.mat',mdict_train)
    print('Simulations complete, data saved to directory.')
    
if __name__ =='__main__':
    try:
        main()
    except Exception as e:
        print(e, '\n')
        print(instructions)