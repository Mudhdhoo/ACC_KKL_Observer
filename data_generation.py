import numpy as np 

# Runge-Kutta 4
def RK4(f, a, b, N, v, inputs):
    h = (b-a) / N
    x = [v]
    t = [a]
    u = 0
        
    for i in range(0,N):
        if inputs != None:
            #u = np.array([inputs(t[-1])])
            u = np.array(inputs(t[-1]))
        k1 = f(u, v)
        k2 = f(u, v + h/2*k1)
        k3 = f(u, v + h/2*k2)
        k4 = f(u, v + h*k3)
        
        v = v + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        x.append(np.ndarray.tolist(v)) 
        
        time = t[-1] + h
        t.append(time)
                 
    return x, np.array(t)

def KKL_observer_data(M, K, y, a, b, ic, N):
    scalar_y = False
    data = []
    size_z = M.shape[0]
    h = (b-a) / N
    
    #Check if y is scalar or vector
    if y.ndim > 2:                                    # Reshape y from (m,) --> (m, 1) for matrix multiplication
        f = lambda y,z: np.matmul(M,z) + np.matmul(K, np.expand_dims(y,1))  
    else:
        f = lambda y,z: np.matmul(M,z) + K*y 
        scalar_y = True
        
    for output, init in zip(y, ic):
        x = [np.ndarray.tolist(init)]
        v = np.array(x).T
        if scalar_y == True:
            truncated_output = np.delete(output,0)    # Ignore the first output value as we already have the initial conditions
        else:
            truncated_output = output[1:, :]
        for i in truncated_output:
            k1 = f(i, v)
            k2 = f(i, v + h/2*k1)
            k3 = f(i, v + h/2*k2)
            k4 = f(i, v + h*k3)
        
            v = v + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

            a = np.reshape(v.T, size_z)
            x.append(np.ndarray.tolist(a)) 
        data.append(np.array(x))

    return np.array(data)

def beta_ic(beta, start):
    return beta / np.sqrt(2) + start

def sample_circular(delta:np.ndarray, num_samples:int) -> np.ndarray:
    """
    Sampling of initial conditions from concentric cricles. Returns a 4D array
    of shape (Number of circles, Number of sampeles in each, 1, 2)
    """
    ic = []
    for distance in delta:
        r = distance + np.sqrt(2)
        angles = np.arange(0, 2*np.pi, 2*np.pi / num_samples)
        x = r*np.cos(angles, np.zeros([1, num_samples])).T
        y = r*np.sin(angles, np.zeros([1, num_samples])).T
        init_cond = np.concatenate((x,y), axis = 1)
        ic.append(np.expand_dims(init_cond, axis = 1))
    return np.array(ic)

def sample_spherical(delta:np.ndarray, num_samples:int) -> np.ndarray:
    """
    Sampling of initial conditions from expanding spherical shells. Returns a 4D array of
    shape (Number of spheres, Number of circles in each sphere, Number of points in each circle, 3)
    """
    r = delta + np.sqrt(0.02)
    theta = np.arange(0, 2*np.pi, 2*np.pi/num_samples)
    phi = np.arange(0, np.pi, (np.pi)/num_samples)
    x = lambda r, theta, phi: r*np.cos(theta)*np.sin(phi)
    y = lambda r, theta, phi: r*np.sin(theta)*np.sin(phi)
    z = lambda r, phi: r*np.cos(phi)

    sphere = []
    for radius in r:
        circles = []
        for angle in phi:
            x_coord = x(radius, theta, np.ones(len(theta))*angle).reshape(-1,1)
            y_coord = y(radius, theta, np.ones(len(theta))*angle).reshape(-1,1)
            z_coord =z(radius, np.ones(len(theta))*angle).reshape(-1,1)
            circle_coord = np.concatenate((x_coord, y_coord, z_coord), axis = 1)
            circles.append(circle_coord)
        sphere.append(circles)

    return np.array(sphere) 
            
def calc_neg_t(M:np.ndarray, z_max:int, e:int) -> int:
    w, v = np.linalg.eig(M)
    min_ev = np.min(np.abs(np.real(w)))
    kappa = np.linalg.cond(v)
    s = np.sqrt(z_max*M.shape[0])
    t = 1/min_ev*np.log(e / (kappa*s))
    
    return t
