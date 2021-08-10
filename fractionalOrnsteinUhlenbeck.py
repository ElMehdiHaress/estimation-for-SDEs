import numpy as np
from math import *

def ornstein_uhlenbeck(dt,n,drift,sigma,H):
    T = dt*n
    trials = 1
    B = fBm(T,n,H,trials)
    x = np.zeros((n+1,trials))
    for i in range(n):
        x[i+1,:] = x[i,:] - dt * x[i,:] * drift + sigma*(B[i+1,:]-B[i,:])
    return x[:,0] 
