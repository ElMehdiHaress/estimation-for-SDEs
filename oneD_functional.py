from scipy.special import gamma
import numpy as np
from math import *

def functional_theta(x,theta, param, arg1, arg2):
    if param = 'H':
        diag = (arg2**2)*(theta*gamma(2*theta) )*arg1**(-2*theta)
        y = np.random.normal(0,sqrt(diag),10000)
        return wassertein1(x[:,0],y)
    if param='drift':
        diag = (arg2**2)*(arg1*gamma(2*arg1) )*theta**(-2*arg1)
        y = np.random.normal(0,sqrt(diag),10000)
        return wassertein1(x[:,0],y)
    if param = 'sigma':
        diag = (theta**2)*(arg2*gamma(2*arg2) )*arg1**(-2*H)
        y = np.random.normal(0,sqrt(diag),10000)
        return wassertein1(x[:,0],y)
