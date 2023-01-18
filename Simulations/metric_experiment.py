import sys
import torch
import Systems
import numpy as np
import scipy.io as spio
import Observer
from Observer import System_z, Observer
from Dataset import DataSet
from data_generation import sample_circular, listdir_filter

instructions = """To run the script, create a path and download the repository contents. Place the trained models
in a seperate directory. Run python metric_experiment.py arg1 arg2 arg3
    arg1 = Location of model directory.
    arg2 = Location to store the data.
    arg3 = Name of the generated MATLAB file."""

def main():
    model_path = sys.argv[1]
    model_names = listdir_filter(model_path)
    save_dir = sys.argv[2]
    file_name = sys.argv[3] 

    # A and B matricies 
    A = np.array([-6.5549,  4.6082, -5.2057, 3.3942, 6.0211,
                  -10.9772, -2.3362, -3.7164, -3.9566, -3.7166,
                  -1.9393, -0.2797, -2.7983, -0.8606, -4.8050,
                  -10.5100, -1.0820, -2.6448, -2.1144, -7.0080,
                  -10.1003, -0.5111, 1.0275, 3.1996, -0.3463]).reshape(5,5)

    B = np.ones([5,1])

    a = 0       # Starting time
    b = 50      # End time
    N = 1000    # RK4 intervals
    num_ic = 50       # Number of initial conditions to be sampled
    limits_normal = np.array([[-1, 1], [-1, 1]])    # Sample space for normal datapoints

    # System setup 
    sys_dim5 = Systems.RevDuff(5, add_noise=False)
    z_sys5 = System_z(A, B, sys_dim5)
    dataset = DataSet(sys_dim5, A, B, a, b, N, num_ic, limits_normal, PINN_sample_mode = 'split traj', data_gen_mode = 'backward sim')

    # Load trained models
    models = []
    for model in model_names:
        models.append(torch.load(model_path+'/'+model))

    # Create observers from models
    observers = []
    for model in models:
        observers.append(Observer(sys_dim5, z_sys5, model, a, b, N, init_z_zero=False))

    delta = np.arange(0,10.5,0.5)
    test_ic = sample_circular(delta, 10)

    # Generate data
    mdict = {'delta':delta}
    for model, observer in zip(model_names, observers):
        met, matrix = observer.calc_gen_metric(dataset.ic, test_ic)
        mdict[model] = met
        mdict['{}_maxtrix'.format(model)] = matrix.T

    # Save to directory
    spio.savemat(save_dir + '/{}'.format(file_name), mdict)

if __name__ == '__main__':
    try:
        print('Beginning simulations.')
        main()
        print('End of program.')
    except Exception as e:
        print(e, '\n')
        print(instructions)
