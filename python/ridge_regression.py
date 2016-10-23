# -*- coding: utf-8 -*-

import numpy as np
from costs import compute_loss_ridge

def ridge_regression(y, tx, lamb):
    """ Implements ridge regression.
       Note: to avoid conflict or inconsistencies between regression methods, 
       a python function compute_loss_ridge has been written explicity and only for ridge regression
       in costs.py"""
    
    # Get w solving Ax = b
    A = np.transpose(tx).dot(tx) + 2*lamb*len(y)*np.identity(tx.shape[1])
    b = np.transpose(tx).dot(y)
    w_opt = np.linalg.solve(A, b)
    
    loss = compute_loss_ridge(y, tx, w_opt)
    
    return loss, w_opt
