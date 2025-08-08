from scipy.special import gamma
import numpy as np
from math import *
from sde_estim.distance import wassertein1

#We assume here and in the rest of the code an order for the parameters of the fractional Ornstein Uhlenebck model which is: drift parameter, hurst parameter and
#finally the diffusion parameter




def functional_theta(x,theta, param, arg1, arg2):
    '''
    Computes the Wassertein distance between the observed sample and the invariant measure
    
    args:
        x:     sample (array)
        theta: the parameter we want to estimate (float)
        param: the parameter we want to estimate (string)
        arg1:  first known parameter
        arg2:  second known parameter
    ''' 
    if param == 'H':
        diag = (arg2**2)*(theta*gamma(2*theta) )*arg1**(-2*theta)
        y = np.random.normal(0,sqrt(diag),10000)
        return wassertein1(x[:,0],y)
    if param == 'drift':
        diag = (arg2**2)*(arg1*gamma(2*arg1) )*theta**(-2*arg1)
        y = np.random.normal(0,sqrt(diag),10000)
        return wassertein1(x[:,0],y)
    if param == 'sigma':
        diag = (theta**2)*(arg2*gamma(2*arg2) )*arg1**(-2*H)
        y = np.random.normal(0,sqrt(diag),10000)
        return wassertein1(x[:,0],y)



__all__= ["functional_theta"]
