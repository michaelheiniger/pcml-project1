# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.zeros((x.shape[0], degree+1))
    for d in range(0,degree+1):
        phi[:, d] = np.power(x,d)
    return phi

def build_poly_matrix(tx, degree):
    """polynomial basis function for input data x, 
    the returned matrix is of the form [1 x^1 ... x^(degree)]"""
    rows = tx.shape[0]
    cols = int(tx.size / rows)
    dummy = np.copy(tx)
    
    if cols == 1: #add a second dimension to 1D array
        dummy = np.reshape(dummy, (rows,1)) 
    #fill phi with the powers of tx
    phi = np.empty((rows, cols * degree + 1))
   
    # Add a column of one at the first position (offset)
    phi[:,0] = np.ones((1, rows))
    for d in range(0, degree):
        phi[:, 1+d*cols: 1+(d+1) * cols] = np.power(dummy, d+1)
    return phi