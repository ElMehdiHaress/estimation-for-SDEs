import numpy as np
from math import *
from fractionalBrownianmotion import fBm

def ornstein_uhlenbeck(dt,n,drift,sigma,H):
    '''
    Generates a path of fractional Ornstein Uhlenbeck using a Euler scheme with timestep dt
    
    args:
        dt:    timestep for the Euler scheme (float)
        n:     sample size (array)
        drift:      drift parameter (float)
        sigma:      diffusion parameter (float)
        H:          hurst parameter (float)
    '''    
    T = dt*n
    trials = 1
    B = fBm(T,n,H,trials)
    x = np.zeros((n+1,trials))
    for i in range(n):
        x[i+1,:] = x[i,:] - dt * x[i,:] * drift + sigma*(B[i+1,:]-B[i,:])
    return x[:,0] 

def increments(x,q):
    '''
    Given a discrete sample path x, generates a q-dimensional sample which contains 
    the original sample and its increments (to the qth order)
    
    args:
        x:    sample (array)
        q:     number of increments desired (float)
        
    returns: (n x q)-array    
    '''    
    n = len(x)-q
    X = np.zeros((n,q))
    for i in range(1,q):
        X[:,i] = x[i:i+n] - x[0:n]
    X[:,0] = x[0:n]    
    return X    


def true_sample(n,q,h,xi,sigma,H):
    '''
    Given an observation timestep h, generates the associated discrete sample path of fractional Ornstein Uhlenbeck.
    
    args:
        n:     sample size (float)
        q:     number of increments desired (float)
        h:     observation timestep
        xi:    drift parameter
        sigma: diffusion parameter
        H:     hurst parameter
        
    returns:  Observed process + its increments ->  (n x q)-array    
    '''   
    dt = 0.001
    x = ornstein_uhlenbeck(dt,int((n+q)*(h/dt)), xi, sigma, H)
    good_indices = [i for i in range(int((n+q)*(h/dt))) if i % int(1/h) == 0]
    x = np.array([x[i] for i in good_indices])
    X = increments(x,q)
    return X

def euler_sample(n,q,gamma,xi,sigma,H):
    x = ornstein_uhlenbeck(gamma,n+q, xi, sigma, H)
    X = increments(x,q)
    return X
