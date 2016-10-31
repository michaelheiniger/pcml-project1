import numpy as np
import matplotlib.pyplot as plt
from plots import error_rate_visualization_ridge
from helpers import build_k_indices
from proj1_helpers import predict_labels_LR
from implementations import reg_logistic_regression, compute_log_likelihood_penalized


def cross_validation(y, x, k_indices, k, lambda_, initial_w, gamma, max_iters):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train    
    indices_for_test = k_indices[k]
    x_test, y_test = x[indices_for_test], y[indices_for_test]
    x_training, y_training = np.delete(x, indices_for_test, axis=0), np.delete(y, indices_for_test, axis=0)

    # ridge regression 
    w_opt, loss_tr = reg_logistic_regression(y_training, x_training, lambda_, initial_w, max_iters, gamma)

    # calculate the loss for test data
    loss_te = compute_log_likelihood_penalized(y_test, x_test, w_opt, lambda_)
    
    y_pred = predict_labels_LR(w_opt, x_test)
    y_pred[y_pred == -1] = 0
    
    error_rate = (y_pred != y_test).sum()/len(y_pred)

    return loss_tr, loss_te, error_rate


def run_reg_logistic_regression(y, x, initial_w ,gamma, max_iters, k_fold=10,lambdas=np.logspace(-10, 10, 30), 
                                seed=1, filename="bias_var_decom_RLR"):
    """ Perform Regularized Logistic regression using k-fold cross-validation and plot error rate.
    By default, the seed is 1 and the whole cross-validation process is done only 
    once and the result is then plotted. Return the error rate, training and test loss for every lambda.
    """
    if k_fold <= 1:
        raise ValueError('The value of k_fold must be larger or equal to 2.')

    np.random.seed(seed)

    k_indices = build_k_indices(y, k_fold, seed)

    loss_tr = np.zeros((k_fold, len(lambdas)))
    loss_te = np.zeros((k_fold, len(lambdas)))
    error_rate = np.zeros((k_fold, len(lambdas)))

    # K-fold cross-validation:
    for k in range(0, k_fold):
        for index_lambda, lambda_ in enumerate(lambdas):
            loss_tr[k, index_lambda], loss_te[k, index_lambda], error_rate[k, index_lambda] = cross_validation(y, x, k_indices, k, lambda_, initial_w, gamma, max_iters)

    # Plot the error rate for every lambda
    error_rate_visualization_ridge(lambdas, error_rate, filename)
    return error_rate, loss_te, loss_tr
