""" Functions implementing different regression models."""

import numpy as np
import gradient_descent as gd
import stochastic_gradient_descent as sgd

def least_squares_GD(y, tx, gamma, max_iters):
    """ Linear Regression using Gradient Descent"""
    
    # initialization of the weights with zeros
    w_initial = np.array(np.zeros(tx.shape[1]))
    # get the losses and weights for every iteration
    gd_losses, gd_ws = gd.gradient_descent(y, tx, w_initial, max_iters, gamma)
    # take the best values
    opt_weights = gd_ws[max_iters-1]
    opt_loss = gd_losses[max_iters-1]
    
    return opt_weights, opt_loss



def least_squares_SGD(y, tx, gamma, max_iters):
    """Linear Regression using Stochastic Gradient Descent"""
    
    # initalization of the weights with zeros
    w_initial = np.array(np.zeros(tx.shape[1]))
    # get the losses and weights for every iteration
    sgd_losses, sgd_ws = sgd.stochastic_gradient_descent(y, tx, w_initial, batch_size, max_iters, gamma)
    # take the best values
    opt_weights = sgd_ws[max_iters-1]
    opt_loss = sgd_losses[max_iters-1]
    
    return opt_weights, opt_loss



def least_squares(y, tx):
    # TODO Sandro
    raise NotImplementedError

def ridge_regression(y, tx, lambda_):
    # TODO Sandro
    raise NotImplementedError

def logistic_regression(y, tx, gamma, max_iters):
    # TODO Michael
    raise NotImplementedError

def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    # TODO Michael
    raise NotImplementedError





