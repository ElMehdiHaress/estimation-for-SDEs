"""
Distance tools (Wasserstein, characteristic-function based, …)
"""

import numpy as np
import sys
!{sys.executable} -m pip install pyemd
from pyemd import emd_samples
def wassertein1(x,y):
    '''
    Computes the Wassertein distance between two samples x and y
    
    args:
        x:     first sample (array)
        y:     second sample (array)
    '''
    return emd_samples(x,y,bins='auto')


__all__=["wassertein1"]
