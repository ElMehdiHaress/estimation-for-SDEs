"""
Simulation utilities for additive fractional SDEs.
Content :
  - davies_harte : one path of fractional Brownian motion using the Davies Harte method
  - fBm : multiple paths of fBm
  - ornstein_uhlenbeck: Generates a path of fractional Ornstein Uhlenbeck using a Euler scheme
  - increments: Given a discrete sample path x, generates a q-dimensional sample which contains 
    the original sample and its increments
  - true_sample: Given an observation timestep h, generates the associated discrete sample path of fractional Ornstein Uhlenbeck.
  - euler_sample: computes an euler scheme for the O-U process 
"""

import numpy as np
from math import *
import matplotlib.pyplot as plt

#Two functions which simulate trajectories of the fractional Brownian motion.

def davies_harte(T, N, H):
    '''
    Generates a sample path of fractional Brownian Motion using the Davies Harte method
    
    args:
        T:      length of time (in years)
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
    g = [gamma(k,H) for k in range(0,N)];    r = g + [0] + g[::-1][0:N-1]

    #Step 1 (eigenvalues)
    j = np.arange(0,2*N);   k = 2*N-1
    lk = np.fft.fft(r*np.exp(2*np.pi*complex(0,1)*k*j*(1/(2*N))))[::-1]

    #Step 2 (get random variables)
    Vj = np.zeros((2*N,2), dtype=np.complex); 
    Vj[0,0] = np.random.standard_normal();  Vj[N,0] = np.random.standard_normal()
    
    for i in range(1,N):
        Vj1 = np.random.standard_normal();    Vj2 = np.random.standard_normal()
        Vj[i][0] = Vj1; Vj[i][1] = Vj2; Vj[2*N-i][0] = Vj1;    Vj[2*N-i][1] = Vj2
    
    #Step 3 (compute Z)
    wk = np.zeros(2*N, dtype=np.complex)   
    wk[0] = np.sqrt((lk[0]/(2*N)))*Vj[0][0];          
    wk[1:N] = np.sqrt(lk[1:N]/(4*N))*((Vj[1:N].T[0]) + (complex(0,1)*Vj[1:N].T[1]))       
    wk[N] = np.sqrt((lk[0]/(2*N)))*Vj[N][0]       
    wk[N+1:2*N] = np.sqrt(lk[N+1:2*N]/(4*N))*(np.flip(Vj[1:N].T[0]) - (complex(0,1)*np.flip(Vj[1:N].T[1])))
    
    Z = np.fft.fft(wk);     fGn = Z[0:N] 
    fBm = np.cumsum(fGn)*(N**(-H))
    fBm = (T**H)*(fBm)
    path = np.array([0] + list(fBm))
    return path

def fBm(T,N,H,trials):
    '''
    Generates multiple sample path of fractional Brownian Motion using the Davies Harte method
    
    args:
        T:      length of time (in years)
        N:      number of time steps within timeframe
        H:      Hurst parameter
        trials: number of paths
    '''
    B = np.zeros((N+1,trials))
    for i in range(trials):
        B[:,i] = davies_harte(T,N,H)
    return B  

#Ornstein-Uhlenbeck simulation:

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


__all__= ["davies_harte","fBm","ornstein_uhlenbeck","increments","true_sample","euler_sample"]
