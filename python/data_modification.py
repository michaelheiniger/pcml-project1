# -*- coding: utf-8 -*-
"""regrouping some data modification methods"""

import numpy as np


def replace_by_mean(tX):
    """replaces the -999. values by the mean of the respective column"""
    
    new_tX = np.copy(tX)
    for col in range(0, tX.shape[1]):
        
        #compute the mean over all 'normal' entries
        idx = (tX[:, col] != -999.)
        mean_col = np.mean(tX[:, col][idx])
        #replace the -999. by the mean
        new_tX[:, col][np.logical_not(idx)] = mean_col
    
    return new_tX