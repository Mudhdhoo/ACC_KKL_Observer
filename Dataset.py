import torch
import data_generation as data
import numpy as np
from Systems import System

class DataSet(torch.utils.data.Dataset):
    """
    Dataset class to generate synthetic x, y and z data.
    The set is split up into data for normal loss and data for physics loss.

    ---------------- Parameters ----------------
    system: Systems
        A system instance created from classes within Systems.py

    M: ndarray
        M matrix in the z dynamics.

    K: ndarray
        K matrix in the z dynamics.

    a: int
        Start time of ODE solver.

    b: int
        End time of ODE solver.
    
    N: int
        Number of intervals in [a,b] with step size (b-a)/N

    samples: int
        Numer of initial conditions to be sampled for data generation.

    limits_normal: ndarray
        Limits on the state sample space of initial conditions used to generate
        data to compute the normal loss.

    PINN_sample_mode: int
        Either 'split set' or 'split traj. 
        If 'split set', the samples for the physics datapoints will be generated from a separate set of
        initial conditions. To set a different limit on the state sample space for the physics
        points, use the set_physics_limit method. 
        If 'split traj', These points will be generated from the same initial conditions,
        with every other sample in a given trajectory being assigned as a physics datapoint.
        Default set to 2.

    data_gen_mode: str
        Either 'negative forward' or 'backward sim'
        If 'negative forward, start simulation from a given negative time untill 0. Use the outputs of
        the simuluation to obtain z(0).
        If 'backward sim', the system is simulated backwards from 0 to a given negative time. The outputs are
        used to simulate z system forward to obtain z(0).
    """

    def __init__(self, system: System, M: np.ndarray, K: np.ndarray, a: int, b: int, N: int , samples: int, limits_normal: np.ndarray, PINN_sample_mode: str = 'split traj', data_gen_mode: str = 'negative forward' ) -> None:
        super().__init__()
        self.M = M
        self.K = K
        self.system = system
        self.a = a
        self.b = b
        self.N = N
        self.samples = samples
        self.limits_normal = limits_normal
        self.PINN_sample_mode = PINN_sample_mode
        self.data_gen_mode = data_gen_mode
        self.train_data = self.generate_data(seed = 888)    # Generate synthetic data x, z, y
        self.data_length = self.train_data[0].shape[0]*self.train_data[0].shape[1]         # Total number of samples
        x = torch.from_numpy(self.train_data[0]).view(self.data_length, system.x_size)  # Convert to tensors and reshape
        z = torch.from_numpy(self.train_data[1]).view(self.data_length, system.z_size)
        # Check if output is vector or scalar
        if self.train_data[2].ndim > 2:
            y = torch.from_numpy(self.train_data[2]).view(self.train_data[2].shape[0]*self.train_data[2].shape[1], self.train_data[2].shape[2])     # y data
        else:
            y = torch.from_numpy(self.train_data[2]).view(self.train_data[2].shape[0]*self.train_data[2].shape[1])

        if PINN_sample_mode == 'split set':
            # ----------------------- Normal loss data ----------------------- 
            self.x_data = x
            self.z_data = z
            self.output_data = y
            self.ic_normal = self.train_data[4]
            #-----------------------------------------------------------------

            # ----------------------- Physics loss data ----------------------- 
            self.train_data_ph = self.generate_data(seed = 8888)
            self.data_length_ph = self.train_data_ph[0].shape[0]*self.train_data_ph[0].shape[1]         # Total number of samples
            self.x_data_ph = torch.from_numpy(self.train_data_ph[0]).view(self.data_length_ph, system.x_size)
            self.z_data_ph = torch.from_numpy(self.train_data_ph[1]).view(self.data_length_ph, system.z_size)
            self.ic_ph = self.train_data_ph[4]
            # Check if output is vector or scalar    
            if self.train_data_ph[2].ndim > 2:
                self.output_data_ph = torch.from_numpy(self.train_data_ph[2]).view(self.train_data_ph[2].shape[0]*self.train_data_ph[2].shape[1], self.train_data_ph[2].shape[2])     # y data
            else:
                self.output_data_ph = torch.from_numpy(self.train_data_ph[2]).view(self.train_data_ph[2].shape[0]*self.train_data_ph[2].shape[1])
            #--------------------------------------------------------------
        elif PINN_sample_mode == 'split traj':
            self.data_length = int(self.data_length / 2)
            self.x_data = x[::2]
            self.z_data = z[::2]
            self.output_data = y[::2]
            self.x_data_ph = x[1::2]
            self.z_data_ph = z[1::2]
            self.output_data_ph = y[1::2]
            self.ic = self.train_data[4]
        elif PINN_sample_mode == 'no physics':
            self.x_data = x
            self.z_data = z
            self.output_data = y
            self.x_data_ph = x
            self.z_data_ph = y
            self.output_data_ph = y
            self.ic = self.train_data[4]
        else:
            raise Exception('Sample mode must be either ''split set'', ''split traj'' or ''no physics''.')

        # ----------------------- Mean and standard deviation -----------------------     
        self.mean_x = torch.mean(self.x_data, dim = 0)
        self.mean_z = torch.mean(self.z_data, dim = 0)
        self.mean_output = torch.mean(self.output_data, dim = 0)
        self.std_x = torch.std(self.x_data, dim = 0)
        self.std_z = torch.std(self.z_data, dim = 0)
        self.std_output = torch.std(self.output_data, dim = 0)

        self.mean_x_ph = torch.mean(self.x_data_ph, dim = 0)
        self.mean_z_ph = torch.mean(self.z_data_ph, dim = 0)
        self.mean_output_ph = torch.mean(self.output_data_ph, dim = 0)
        self.std_x_ph = torch.std(self.x_data_ph, dim = 0)
        self.std_z_ph = torch.std(self.z_data_ph, dim = 0)
        self.std_output_ph = torch.std(self.output_data, dim = 0)
        #-----------------------------------------------------------------------------
        self.time = torch.from_numpy(self.train_data[3])
        
    def __len__(self) -> None:
        return self.data_length
    
    def generate_data(self, seed: int) -> tuple:
        """
        System initial conditions are sampled with LHS. Data is generated by simulating
        x and z systems. 
        """

        t_back = data.calc_neg_t(self.M, 10, 1e-6)

        h = (self.b-self.a)/self.N
        ic = self.system.sample_ic(self.limits_normal, self.samples, seed = seed)     # Sample initial conditions for x
        x_data_fw, output_fw, t_fw = self.system.generate_data(ic, self.a, self.b, self.N)      # Forward simulation 

        if self.data_gen_mode == 'backward sim':
            _ , output_bw, t_bw = self.system.generate_data(ic, self.a, t_back, int(np.abs(t_back/h)))      # Backward simulation
            output_bw = np.flip(output_bw, axis = 1)
        elif self.data_gen_mode == 'negative forward':
            _ , output_bw, t_bw = self.system.generate_data(ic, t_back, self.a, int(np.abs(t_back/h)))      # Backward simulation
        else:
            raise Exception('Data generation mode must be either ''backward sim'' or ''negative forward''.')

        ic_z_bw = np.random.rand(self.samples, self.system.z_size)
        z_data_fw1 = data.KKL_observer_data(self.M, self.K, output_bw, t_back, self.a, ic_z_bw, int(np.abs(t_back/h)))    
        ic_z = z_data_fw1[:, -1, :]     # The last element of each trajectory
        #ic_z = np.zeros([samples, self.system.z_size])
        z_data_fw2 = data.KKL_observer_data(self.M, self.K, output_fw, self.a, self.b, ic_z, self.N)

        return x_data_fw, z_data_fw2, output_fw, t_fw, ic

    def set_physics_limits(self, limits: np.ndarray) -> None:
        """
        Sets the limits of the state sample space for the physics datapoints.
        Can only be used if sample_mode is 2.
        """
        if self.PINN_sample_mode == 'split traj':
            self.limit_physics = limits
        else:
            raise Exception('Can only set limits if sample mode is 1.')  
    
    def normalize(self) -> None:
        """
        Old method to normalize all the data before training.
        Use the normalizer class instead.
        """
        self.x_data = (self.x_data - self.mean_x) / self.std_x
        self.z_data = (self.z_data - self.mean_z) / self.std_z
        self.output_data = (self.output_data - self.mean_output) / self.std_output       
        
    def __getitem__(self, idx: int) -> None:
        x = self.x_data[idx]
        z = self.z_data[idx]
        y = self.output_data[idx]
        x_ph = self.x_data_ph[idx]
        y_ph = self.output_data_ph[idx]
        return [x.float(), z.float(), y.float(), x_ph.float(), y_ph.float()]