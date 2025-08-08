from scipy.special import psi
from scipy.special import gamma
import scipy.integrate as integrate
import numpy as np
from math import *

#When estimating two parameters, we look at the process and its first increments, which gives us a new two dimensional process. In the Ornstein Uhlenbeck case,
#the invariant distribution of this process is Gaussian. We first show how to compute its covariant matrix, and the derivatives of this matrix with respect to the
#parameters. Then we show how to compute the gradient of the distance between this new sample and its invariant distribution.


def cov_matrix(h,xi,H,sigma):
    '''
    Computes the covariant matrix
    args:
        xi: drift parameter (float)
        H: hurst parameter (float)
        sigma: diffusion parameter (float)
    returns: cov (2x2 matrix)    
    '''
    I = integrate.quad(lambda x: cos(h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I = I[0]
    non_diag = (sigma**2)*gamma(2*H+1)*sin(pi*H)*I/pi  - (sigma**2)*(H*gamma(2*H) )*xi**(-2*H)
    diag_1 = (sigma**2)*(H*gamma(2*H) )*xi**(-2*H)
    diag_2 = 2*(sigma**2)*(H*gamma(2*H) )*xi**(-2*H)-2*(sigma**2)*gamma(2*H+1)*sin(pi*H)*I/pi  
    cov =  [[diag_1, non_diag],[non_diag,diag_2]]
    return cov

def partial_cov(h,xi,H,sigma):
    '''
    Computes the partial derivatives of the covariant matrix
    args:
        xi: drift parameter (float)
        H: hurst parameter (float)
        sigma: diffusion parameter (float)
    returns: [cov_xi, cov_H, cov_sigma]  (where cov_X is a 2x2 matrix, i.e the partial derivative with respect to X)    
    '''
    I = integrate.quad(lambda x: cos(h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I = I[0]
    non_diag = 2*(sigma)*gamma(2*H+1)*sin(pi*H)*I/pi  - (2*sigma)*(H*gamma(2*H) )*xi**(-2*H)
    diag_1 = (2*sigma)*(H*gamma(2*H) )*xi**(-2*H)
    diag_2 = 2*(2*sigma)*(H*gamma(2*H) )*xi**(-2*H)-2*(2*sigma)*gamma(2*H+1)*sin(pi*H)*I/pi  
    cov_sigma =  [[diag_1, non_diag],[non_diag,diag_2]]
    

    I = integrate.quad(lambda x: -cos(h*x)*(x**(1-2*H))*(2*xi)/(xi**2 + x**2)**2, 0, 1000)
    I = I[0]
    non_diag = (sigma**2)*gamma(2*H+1)*sin(pi*H)*I/pi  - (-2*H)*(sigma**2)*(H*gamma(2*H) )*xi**(-2*H-1)
    diag_1 = (-2*H)*(sigma**2)*(H*gamma(2*H) )*xi**(-2*H-1)
    diag_2 = 2*(-2*H)*(sigma**2)*(H*gamma(2*H) )*xi**(-2*H-1)-2*(sigma**2)*gamma(2*H+1)*sin(pi*H)*I/pi  
    cov_xi =  [[diag_1, non_diag],[non_diag,diag_2]]
    
    I = integrate.quad(lambda x: cos(h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I = I[0]
    II = integrate.quad(lambda x: (-2*log(x))*cos(h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    II = II[0]
    non_diag = (sigma**2)*gamma(2*H+1)*cos(pi*H)*I + 2*(sigma**2)*psi(2*H+1)*gamma(2*H+1)*sin(pi*H)*I/pi + (sigma**2)*gamma(2*H+1)*sin(pi*H)*II/pi  - (sigma**2)*(psi(2*H+1)*gamma(2*H+1))*xi**(-2*H) - (-2*log(xi))*(sigma**2)*(gamma(2*H+1)/2)*xi**(-2*H)
    diag_1 = (sigma**2)*(psi(2*H+1)*gamma(2*H+1))*xi**(-2*H) + (-2*log(xi))*(sigma**2)*(gamma(2*H+1)/2)*xi**(-2*H)
    diag_2 = 2*diag_1 -2*( (sigma**2)*gamma(2*H+1)*cos(pi*H)*I + 2*(sigma**2)*psi(2*H+1)*gamma(2*H+1)*sin(pi*H)*I/pi + (sigma**2)*gamma(2*H+1)*sin(pi*H)*II/pi  )
    cov_H =  [[diag_1, non_diag],[non_diag,diag_2]]
    
    return [cov_xi, cov_H, cov_sigma]
  
  
def random_va(p, trials):
    '''
    The distance between the sample and its invariant measure can be written as the expectation of L(X), where L may represent a loss function and X has a certain 
    density function (what we call g_p in our work). Here, we generate many trials of a random variable that has g_p as density.
    args:
        p: specifies the desired density (int)
        trials: number of trials (int)
    '''
    u = np.random.uniform(0,2*pi,trials)
    v = np.random.uniform(0,1,trials)
    v = (v**(1/(1-p)) - 1)**(1/2)
    return [v*np.cos(u), v*np.sin(u)]
  
  
  
def gradient(h,x,theta,pb,p):
    '''
    Computes a 'stochastic' gradient of the dsitance.
    args:
        x: sample (array)
        theta: drift parameter, hurst parameter and diffusion parameter (array)
        pb : 1 if estimating xi and H, 2 if estimating xi and sigma, 3 if estimating H and sigma
        p : specifies the desired density (int)
    returns: 2-array    
    '''
    Eta = random_va(p, 100)
    g = []
    cov_partial = partial_cov(h,theta[0],theta[1],theta[2])
    if pb == 1:
        cov_partial = [cov_partial[0],cov_partial[1]]
    if pb == 2:
        cov_partial = [cov_partial[0],cov_partial[2]]
    if pb == 3:
        cov_partial = [cov_partial[1],cov_partial[2]]
    cov_partial = np.array(cov_partial)    
    
    matrix_cov = np.array(cov_matrix(h,theta[0],theta[1],theta[2]))
    
    for i in range(100):
        eta = [Eta[0][i],Eta[1][i]]
        av = np.mean(np.cos(np.inner(eta,x)))
        r = -0.5*np.inner(eta,matrix_cov.dot(eta))
        r0 = -0.5*np.inner(eta,cov_partial[0].dot(eta))
        r1 = -0.5*np.inner(eta,cov_partial[1].dot(eta))
        g += [ [-2*(av - np.exp(r))*r0 , -2*(av - np.exp(r))*r1]  ]
    g = np.array(g)    
    return np.array([np.mean(g[:,0]),np.mean(g[:,1])]) 

__all__=["cov_matrix","partial_cov","random_va","gradient"]
