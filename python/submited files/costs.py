# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e.dot(e))

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_loss(y, tx, w):
    """Calculate the loss.
        You can calculate the loss using mse or mae.
        """
    e = y - tx.dot(w)
    return calculate_mse(e)
# return calculate_mae(e)

def compute_loss_ridge(y, tx, w):
    """ Calculate the loss for ridge regression:
    The version cost function considered is: 1/(2*N) sum_over_n(y_n-x_n*w)^2 + lambda*norm(w)^2.
    """
    e = y - tx.dot(w)
    N = y.shape[0]
    return 1/(2*N)*(np.transpose(e).dot(e)).sum()