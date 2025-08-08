import numpy as np
from math import *
import matplotlib.pyplot as plt
from sde_estim.simulation import true_sample
from numpy import linalg as LA
from sde_estim.estimators.three_d_procedure import random_va

#As we explained before, when estimating thee parameters we consider the consider the O-U process X, its increments X_{.+h}-X_. and X_{.+2h}-X_. 
#However, some may argue that the last increment should better be replaced by X_{.+h}-2X_. +X_{.-h}, which would correspond to the second order 'derivative'. 
#If we consider the second choice, then naturally the covariant matrix and its partial derivatives we computed before will change. The same goes for the gradient.


def cov_matrixv2(h,xi,H,sigma):
    I1 = integrate.quad(lambda x: cos(h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I1 = I1[0]
    I2 = integrate.quad(lambda x: cos(2*h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I2 = I2[0]
    txt =  (sigma**2)*(H*gamma(2*H) )*xi**(-2*H)  
    hxt = (sigma**2)*gamma(2*H+1)*sin(pi*H)*I1/pi 
    h2xt = (sigma**2)*gamma(2*H+1)*sin(pi*H)*I2/pi
    
    non_diag_1 = hxt  - txt
    non_diag_2 = 2*hxt - 2*txt
    non_diag_3 = 3*txt + h2xt - 4*hxt
    diag_1 = txt
    diag_2 = 2*txt -2*hxt
    diag_3 = 6*txt + 2*h2xt - 8*h2xt
    cov =  [[diag_1, non_diag_1, non_diag_2 ],[non_diag_1,diag_2, non_diag_3 ], [non_diag_2, non_diag_3 ,diag_3]]
    return cov


def partial_covv2(h,xi,H,sigma):
    I1 = integrate.quad(lambda x: cos(h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I1 = I1[0]
    I2 = integrate.quad(lambda x: cos(2*h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I2 = I2[0]
    
    txt =  (2*sigma)*(H*gamma(2*H) )*xi**(-2*H)  
    hxt = 2*(sigma)*gamma(2*H+1)*sin(pi*H)*I1/pi
    h2xt = 2*(sigma)*gamma(2*H+1)*sin(pi*H)*I2/pi
    
    non_diag_1 = hxt  - txt
    non_diag_2 = 2*hxt - 2*txt
    non_diag_3 = 3*txt + h2xt - 4*hxt
    diag_1 = txt
    diag_2 = 2*txt -2*hxt
    diag_3 = 6*txt + 2*h2xt - 8*h2xt
    cov_sigma =  [[diag_1, non_diag_1, non_diag_2 ],[non_diag_1,diag_2, non_diag_3 ], [non_diag_2, non_diag_3 ,diag_3]]
    

    I1 = integrate.quad(lambda x: -cos(h*x)*(x**(1-2*H))*(2*xi)/(xi**2 + x**2)**2, 0, 1000)
    I1 = I1[0]
    I2 = integrate.quad(lambda x: -cos(2*h*x)*(x**(1-2*H))*(2*xi)/(xi**2 + x**2)**2, 0, 1000)
    I2 = I2[0]
    txt =  (-2*H)*(sigma**2)*(H*gamma(2*H) )*xi**(-2*H-1)
    hxt = (sigma**2)*gamma(2*H+1)*sin(pi*H)*I1/pi
    h2xt = (sigma**2)*gamma(2*H+1)*sin(pi*H)*I2/pi
    
    non_diag_1 = hxt  - txt
    non_diag_2 = 2*hxt - 2*txt
    non_diag_3 = 3*txt + h2xt - 4*hxt
    diag_1 = txt
    diag_2 = 2*txt -2*hxt
    diag_3 = 6*txt + 2*h2xt - 8*h2xt
    cov_xi = [[diag_1, non_diag_1, non_diag_2 ],[non_diag_1,diag_2, non_diag_3 ], [non_diag_2, non_diag_3 ,diag_3]]
    
    I1 = integrate.quad(lambda x: cos(h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I1 = I1[0]
    I2 = integrate.quad(lambda x: cos(2*h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I2 = I2[0]
    II1 = integrate.quad(lambda x: (-2*log(x))*cos(h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    II1 = II1[0]
    II2 = integrate.quad(lambda x: (-2*log(x))*cos(2*h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    II2 = II2[0]
    txt =  (sigma**2)*(psi(2*H+1)*gamma(2*H+1))*xi**(-2*H) + (-2*log(xi))*(sigma**2)*(gamma(2*H+1)/2)*xi**(-2*H)
    hxt = (sigma**2)*gamma(2*H+1)*cos(pi*H)*I1 + 2*(sigma**2)*psi(2*H+1)*gamma(2*H+1)*sin(pi*H)*I1/pi + (sigma**2)*gamma(2*H+1)*sin(pi*H)*II1/pi 
    h2xt = (sigma**2)*gamma(2*H+1)*cos(pi*H)*I2 + 2*(sigma**2)*psi(2*H+1)*gamma(2*H+1)*sin(pi*H)*I2/pi + (sigma**2)*gamma(2*H+1)*sin(pi*H)*II2/pi 
    
    non_diag_1 = hxt  - txt
    non_diag_2 = 2*hxt - 2*txt
    non_diag_3 = 3*txt + h2xt - 4*hxt
    diag_1 = txt
    diag_2 = 2*txt -2*hxt
    diag_3 = 6*txt + 2*h2xt - 8*h2xt
    cov_H =  [[diag_1, non_diag_1, non_diag_2 ],[non_diag_1,diag_2, non_diag_3 ], [non_diag_2, non_diag_3 ,diag_3]]
    
    return [cov_xi, cov_H, cov_sigma]
  
  
  
def gradient(h,x,theta):
    '''
    Computes a 'stochastic' gradient of the dsitance.
    args:
        x: sample (array)
        theta: drift parameter, hurst parameter and diffusion parameter (array)
    returns: 3-array    
    '''
    Eta = random_va(100)
    g = []
    cov_partial = partial_covv2(h,theta[0],theta[1],theta[2])
    cov_partial = np.array(cov_partial)    
    matrix_cov = np.array(cov_matrixv2(h,theta[0],theta[1],theta[2]))
    
    for i in range(100):
        eta = [Eta[0][i],Eta[1][i],Eta[2][i]]
        av = np.mean(np.cos(np.inner(eta,x)))
        r = -0.5*np.inner(eta,matrix_cov.dot(eta))
        r0 = -0.5*np.inner(eta,cov_partial[0].dot(eta))
        r1 = -0.5*np.inner(eta,cov_partial[1].dot(eta))
        r2 = -0.5*np.inner(eta,cov_partial[2].dot(eta))
        g += [ [-2*(av - np.exp(r))*r0 , -2*(av - np.exp(r))*r1 , -2*(av - np.exp(r))*r2]  ]
    g = np.array(g)    
    return np.array([np.mean(g[:,0]),np.mean(g[:,1]), np.mean(g[:,2])])  

__all__= ["gradient"]
