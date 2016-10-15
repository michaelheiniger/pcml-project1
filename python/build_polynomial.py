# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.zeros((x.shape[0], degree+1))
    powers = np.arange(degree+1)[np.newaxis]
    phi = np.power(x[:,np.newaxis], powers)    
    return phi