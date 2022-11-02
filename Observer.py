import torch
import numpy as np
import data_generation as data
from torch.autograd.functional import jacobian

class System_z:
    """
    Dynamics of the z-system for both autonomous and non-autonomous cases.
    Autonomous:
        z_dot = Mz(t) + Ky(t)
        
    Non-Autonomous:
        z_dot = Mz(t) + Ky(t) + phi(t, z(t))*(u(t) - u0(t))
        phi(t, z(t)) = dT/dx*g

    """
    def __init__(self, M, K, system):
        self.M = M
        self.K = K
        self.is_autonomous = True if system.input == None else False
        self.y_size = system.y_size

    def z_dynamics(self):
        if self.is_autonomous:
            if self.y_size > 1:
                z_dot = lambda y,z: np.matmul(self.M,z) + np.matmul(self.K, np.expand_dims(y,1))
            else:
                z_dot = lambda y,z: np.matmul(self.M,z) + self.K*y 
        
        else:
            if self.y_size > 1:
                z_dot = lambda y,q,z: np.matmul(self.M,z) + np.matmul(self.K, np.expand_dims(y,1)) + q
            else:
                z_dot = lambda y,q,z: np.matmul(self.M,z) + self.K*y + q 

        return z_dot

class Observer:
    def __init__(self, system, z_system, net, a ,b ,N, init_z_zero = True):
        self.system = system
        self.z_system = z_system
        self.f = z_system.z_dynamics()
        self.T = net.net1
        self.T_inv = net.net2
        self.a = a
        self.b = b
        self.N = N
        self.init_z_zero = init_z_zero
    
    def simulate_NA(self, ic, u0, g):
        """
        Online simulation of observer for non-autonomous input-affine systems by the following steps:
        1. Generate y data from the system with input u.
        2. Use the y data to simulate observer dynamics:
            z_dot = Mz(t) + Ky(t) + phi(t, z(t))*(u(t) - u0(t))
            phi(t, z(t)) = dT/dx*g
        3. Use T_inv, the inverse of T to simulate:
            x_hat = T_inv(t, z)

        """
        x, y, t = self.system.generate_data(ic, self.a, self.b, self.N)     # Generate y data
        x = torch.from_numpy(np.reshape(x, (self.N+1,self.system.x_size)))
        u = self.system.input
        size_z = self.z_system.M.shape[0]
        h = (self.b-self.a) / self.N

        z = [[0]*size_z]
        v = np.array(z).T

        y = np.squeeze(y)
        if y.ndim > 2:
            y = y[1:, :]
        else:
            y = np.delete(y,0)

        for idx, output in enumerate(y):
            with torch.no_grad():
                x_hat = self.T_inv(torch.tensor(z[-1]).float())
            u_sub_u0 = u(t[idx]) - u0(t[idx])
            dTdx = jacobian(self.T, x_hat).numpy()
            dTdx_mul_g = np.matmul(dTdx, g)

            q = dTdx_mul_g*u_sub_u0     # phi*(u(t) - u0(t))

            k1 = self.f(output,q, v)
            k2 = self.f(output,q, v + h/2*k1)
            k3 = self.f(output,q, v + h/2*k2)
            k4 = self.f(output,q, v + h*k3)
            
            v = v + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

            a = np.reshape(v.T, size_z)
            z.append(np.ndarray.tolist(a)) 

        z = np.array(z)
        z = torch.from_numpy(z).float()
        with torch.no_grad():
            x_hat = self.T_inv(z)

        error = torch.abs(x-x_hat)

        return x, x_hat, t, error

    def simulate(self, ic, noise_mean = 0, noise_std = 0.3, add_noise = False):
        """
        Online simulation of observer for autonomous systems by the following steps:
        1. Generate y data from system.
        2. Use the y data to simulate observer dynamics:
            z_dot = Mz(t) + Ky(t)
        3. Use T_inv, the inverse of T to simulate:
            x_hat = T_inv(t, z)

        """
        x, y, t = self.system.generate_data(ic, self.a, self.b, self.N)
        x = torch.from_numpy(np.reshape(x, (self.N+1,self.system.x_size)))
        if add_noise:
            np.random.seed(123)
            noise = np.random.normal(noise_mean, noise_std, y.shape)    # Adding Noise
           # noise = np.random.normal(0, 0.05, y.shape)    # Adding Noise
            y = y + noise
        
        if self.init_z_zero:
            ic_z = np.zeros([1,self.system.z_size])
        else:
            with torch.no_grad():
                ic_z = self.T(torch.from_numpy(ic)).numpy()

        z = data.KKL_observer_data(self.z_system.M, self.z_system.K, y, self.a, self.b, ic_z, self.N)
        z = torch.from_numpy(z).view(self.N+1,self.system.z_size).float()

        with torch.no_grad():
            x_hat = self.T_inv(z)

        error = torch.abs(x-x_hat)

        return x, x_hat, t, error
    
    def sim_multi(self, ic_samples, add_noise = False):
        avr_error = 0
        errors = []
        x_traj = []
        x_hat_traj = []
        for idx, ic in enumerate(ic_samples):
            x, x_hat, time, error = self.simulate(ic, add_noise=add_noise)
            avr_error += error
            errors.append(error.numpy())
            x_traj.append(x.numpy())
            x_hat_traj.append(x_hat.numpy())
        avr_error = avr_error / idx
        
        return np.array(x_traj), np.array(x_hat_traj), np.array(errors), avr_error, time

    def calc_gen_metric(self, train_ic, test_ic):
        GE = []
        GE_matrix = []
        p = len(train_ic)
        tau = self.N
        train_ic = np.expand_dims(train_ic, axis = 1)
        x_train, x_hat_train, _, _, time1 = self.sim_multi(train_ic)

        train_error = 0
        for true, est in zip(x_train, x_hat_train):
            sum = 0
            for x, x_hat in zip(true, est):
                error_norm = np.linalg.norm(x-x_hat)**2
                true_norm = np.linalg.norm(x)**2
                sum += error_norm/true_norm
            train_error += sum / tau
    
        train_error_av = train_error / p

        if self.system.x_size == 2:
            for circle in test_ic:
                x_test, x_hat_test, _, _, time2 = self.sim_multi(circle)
                sum_circle = 0
                GE_matrix_col = []
                for true, est in zip(x_test, x_hat_test):
                    error_norm = np.linalg.norm(true - est, axis = 1)**2
                    true_norm = np.linalg.norm(true, axis = 1)**2
                    trajectory_average = np.sum(error_norm / true_norm) / tau
                    sum_circle += trajectory_average
                    GE_matrix_col.append(trajectory_average)
                metric = np.abs((sum_circle / len(circle)) - train_error_av)
                GE.append(metric)
                GE_matrix.append(GE_matrix_col)
        elif self.system.x_size == 3:
            for sphere in test_ic:
                sum_sphere = 0
                GE_matrix_col = []
                ic = sphere.reshape(-1,1,3)
                x_test, x_hat_test, _, _, _ = self.sim_multi(ic)
                for true, est in zip(x_test, x_hat_test):
                    error_norm = np.linalg.norm(true - est, axis = 1)**2
                    true_norm = np.linalg.norm(true, axis = 1)**2
                    trajectory_average = np.sum(error_norm / true_norm) / tau
                    sum_sphere += trajectory_average
                    GE_matrix_col.append(trajectory_average)
                metric = np.abs((sum_sphere / len(ic)) - train_error_av)
                GE.append(metric)
                GE_matrix.append(GE_matrix_col)
                    
        return GE, np.array(GE_matrix)


        
