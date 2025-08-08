from scipy.special import psi
from tqdm import tqdm
from scipy.special import gamma
import scipy.integrate as integrate
import numpy as np
from math import *
from scipy.stats import rv_continuous

#When estimating three parameters, we look at the process and its first and second increments, which gives us a new three dimensional process. In the Ornstein Uhlenbeck case,
#the invariant distribution of this process is Gaussian. We first show how to compute its covariant matrix, and the derivatives of this matrix with respect to the
#parameters. Then we show how to compute the gradient of the distance between this new sample and its invariant distribution.

  
  def cov_matrix(xi,H,sigma):
    I1 = integrate.quad(lambda x: cos(h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I1 = I1[0]
    I2 = integrate.quad(lambda x: cos(2*h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I2 = I2[0]
    non_diag_1 = (sigma**2)*gamma(2*H+1)*sin(pi*H)*I1/pi  - (sigma**2)*(H*gamma(2*H) )*xi**(-2*H)
    non_diag_2 = (sigma**2)*gamma(2*H+1)*sin(pi*H)*I2/pi  - (sigma**2)*(H*gamma(2*H) )*xi**(-2*H)
    diag_1 = (sigma**2)*(H*gamma(2*H) )*xi**(-2*H)
    diag_2 = 2*(sigma**2)*(H*gamma(2*H) )*xi**(-2*H)-2*(sigma**2)*gamma(2*H+1)*sin(pi*H)*I1/pi  
    diag_3 = 2*(sigma**2)*(H*gamma(2*H) )*xi**(-2*H)-2*(sigma**2)*gamma(2*H+1)*sin(pi*H)*I2/pi 
    cov =  [[diag_1, non_diag_1, non_diag_2 ],[non_diag_1,diag_2, -non_diag_2 ], [non_diag_2, -non_diag_2 ,diag_3]]
    return cov


def partial_cov(xi,H,sigma):
    I1 = integrate.quad(lambda x: cos(h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I1 = I1[0]
    I2 = integrate.quad(lambda x: cos(2*h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I2 = I2[0]
    non_diag_1 = 2*(sigma)*gamma(2*H+1)*sin(pi*H)*I1/pi  - (2*sigma)*(H*gamma(2*H) )*xi**(-2*H)
    non_diag_2 = 2*(sigma)*gamma(2*H+1)*sin(pi*H)*I2/pi  - (2*sigma)*(H*gamma(2*H) )*xi**(-2*H)
    diag_1 = (2*sigma)*(H*gamma(2*H) )*xi**(-2*H)
    diag_2 = 2*(2*sigma)*(H*gamma(2*H) )*xi**(-2*H)-2*(2*sigma)*gamma(2*H+1)*sin(pi*H)*I1/pi  
    diag_3 = 2*(2*sigma)*(H*gamma(2*H) )*xi**(-2*H)-2*(2*sigma)*gamma(2*H+1)*sin(pi*H)*I2/pi  
    cov_sigma =  [[diag_1, non_diag_1, non_diag_2 ],[non_diag_1,diag_2, -non_diag_2 ], [non_diag_2, -non_diag_2 ,diag_3]]
    

    I1 = integrate.quad(lambda x: -cos(h*x)*(x**(1-2*H))*(2*xi)/(xi**2 + x**2)**2, 0, 1000)
    I1 = I1[0]
    I2 = integrate.quad(lambda x: -cos(2*h*x)*(x**(1-2*H))*(2*xi)/(xi**2 + x**2)**2, 0, 1000)
    I2 = I2[0]
    non_diag_1 = (sigma**2)*gamma(2*H+1)*sin(pi*H)*I1/pi  - (-2*H)*(sigma**2)*(H*gamma(2*H) )*xi**(-2*H-1)
    non_diag_2 = (sigma**2)*gamma(2*H+1)*sin(pi*H)*I2/pi  - (-2*H)*(sigma**2)*(H*gamma(2*H) )*xi**(-2*H-1)
    diag_1 = (-2*H)*(sigma**2)*(H*gamma(2*H) )*xi**(-2*H-1)
    diag_2 = 2*(-2*H)*(sigma**2)*(H*gamma(2*H) )*xi**(-2*H-1)-2*(sigma**2)*gamma(2*H+1)*sin(pi*H)*I1/pi 
    diag_3 = 2*(-2*H)*(sigma**2)*(H*gamma(2*H) )*xi**(-2*H-1)-2*(sigma**2)*gamma(2*H+1)*sin(pi*H)*I2/pi 
    cov_xi = [[diag_1, non_diag_1, non_diag_2 ],[non_diag_1,diag_2, -non_diag_2 ], [non_diag_2, -non_diag_2 ,diag_3]]
    
    I1 = integrate.quad(lambda x: cos(h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I1 = I1[0]
    I2 = integrate.quad(lambda x: cos(2*h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    I2 = I2[0]
    II1 = integrate.quad(lambda x: (-2*log(x))*cos(h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    II1 = II1[0]
    II2 = integrate.quad(lambda x: (-2*log(x))*cos(2*h*x)*(x**(1-2*H))/(xi**2 + x**2), 0, 1000)
    II2 = II2[0]
    non_diag_1 = (sigma**2)*gamma(2*H+1)*cos(pi*H)*I1 + 2*(sigma**2)*psi(2*H+1)*gamma(2*H+1)*sin(pi*H)*I1/pi + (sigma**2)*gamma(2*H+1)*sin(pi*H)*II1/pi  - (sigma**2)*(psi(2*H+1)*gamma(2*H+1))*xi**(-2*H) - (-2*log(xi))*(sigma**2)*(gamma(2*H+1)/2)*xi**(-2*H)
    non_diag_2 = (sigma**2)*gamma(2*H+1)*cos(pi*H)*I2 + 2*(sigma**2)*psi(2*H+1)*gamma(2*H+1)*sin(pi*H)*I2/pi + (sigma**2)*gamma(2*H+1)*sin(pi*H)*II2/pi  - (sigma**2)*(psi(2*H+1)*gamma(2*H+1))*xi**(-2*H) - (-2*log(xi))*(sigma**2)*(gamma(2*H+1)/2)*xi**(-2*H)
    diag_1 = (sigma**2)*(psi(2*H+1)*gamma(2*H+1))*xi**(-2*H) + (-2*log(xi))*(sigma**2)*(gamma(2*H+1)/2)*xi**(-2*H)
    diag_2 = 2*diag_1 -2*( (sigma**2)*gamma(2*H+1)*cos(pi*H)*I1 + 2*(sigma**2)*psi(2*H+1)*gamma(2*H+1)*sin(pi*H)*I1/pi + (sigma**2)*gamma(2*H+1)*sin(pi*H)*II1/pi  )
    diag_3 = 2*diag_1 -2*( (sigma**2)*gamma(2*H+1)*cos(pi*H)*I2 + 2*(sigma**2)*psi(2*H+1)*gamma(2*H+1)*sin(pi*H)*I2/pi + (sigma**2)*gamma(2*H+1)*sin(pi*H)*II2/pi  )
    cov_H =  [[diag_1, non_diag_1, non_diag_2 ],[non_diag_1,diag_2, -non_diag_2 ], [non_diag_2, -non_diag_2 ,diag_3]]
    
    return [cov_xi, cov_H, cov_sigma]
  
  



I = integrate.quad(lambda x: (x**2)*(1+x**2)**(-2), 0, np.inf)
I = 1/I[0]
class gp_gen(rv_continuous):
    "gp distribution"
    def _pdf(self, x):
        return I*(x**2)*(1+x**2)*(-2)
gp = gp_gen(name='gp')

def random_va(trials):
    '''
    The distance between the sample and its invariant measure can be written as the expectation of L(X), where L may represent a loss function and X has a certain 
    density function (what we call g_p in our work). Here, we generate many trials of a random variable that has g_p as density.
    args:
        trials: number of trials (int)
    '''
    r = gp.rvs(size=trials)
    v = np.arccos(1-2*np.random.uniform(0,1,trials))
    u = np.random.uniform(0,2*pi,trials)
    return [r*np.cos(u)*np.sin(v), r*np.sin(u)*np.sin(v), r*np.cos(v)]
  
  
  
 def gradient(x,theta):
        '''
    Computes a 'stochastic' gradient of the dsitance.
    args:
        x: sample (array)
        theta: drift parameter, hurst parameter and diffusion parameter (array)
    returns: 3-array    
    '''
    Eta = random_va(100)
    g = []
    cov_partial = partial_cov(theta[0],theta[1],theta[2])
    cov_partial = np.array(cov_partial)    
    matrix_cov = np.array(cov_matrix(theta[0],theta[1],theta[2]))
    
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
