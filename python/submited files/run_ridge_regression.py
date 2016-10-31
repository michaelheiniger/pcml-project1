import numpy as np
from costs import compute_loss_ridge
from implementations import ridge_regression
from helpers import *
from proj1_helpers import *
from plots import *

def cross_validation(y, x, k_indices, k, lambda_):
    """ Returns the loss and error rate of ridge regression. """
    
    # Get k'th subgroup for test set, others for training set
    indices_for_test = k_indices[k]
    x_test, y_test = x[indices_for_test], y[indices_for_test]
    x_training, y_training = np.delete(x, indices_for_test, axis=0), np.delete(y, indices_for_test, axis=0)

    # Perform Ridge regression on training set
    w_opt, loss_tr = ridge_regression(y_training, x_training, lambda_)

    # Calculate the loss for test data
    loss_te = compute_loss_ridge(y_test, x_test, w_opt)
    
    # Compute RMSE for training and test set
    rmse_tr = np.sqrt(2 * loss_tr)
    rmse_te = np.sqrt(2 * loss_te)
    
    # Compute the predictions for the test set using the weights of the model
    y_pred = predict_labels(w_opt, x_test)
    
    # Compute the error rate of the predictions
    error_rate = (y_pred != y_test).sum()/len(y_pred)

    return rmse_tr, rmse_te, error_rate

def run_ridge_regression(y, x, lambdas=np.logspace(-10, 10, 30), k_fold=10, seed=1, filename="ridge_errorrate_vs_lambda"):
    """ Perform Ridge regression using k-fold cross-validation and plot the error rate versus lambda. """
    if k_fold <= 1:
        raise ValueError('The value of k_fold must be larger or equal to 2.')

    k_indices = build_k_indices(y, k_fold, seed)

    # Save training/test RMSE and error rate for each iteration of cross-validation
    rmse_tr = np.zeros((k_fold, len(lambdas)))
    rmse_te = np.zeros((k_fold, len(lambdas)))
    error_rate = np.zeros((k_fold, len(lambdas)))

    # K-fold cross-validation:
    for k in range(0, k_fold):
        for index_lambda, lambda_ in enumerate(lambdas):
            rmse_tr[k, index_lambda], rmse_te[k, index_lambda], error_rate[k, index_lambda] = cross_validation(y, x, k_indices, k, lambda_)

    # Plot the error rate versus lambda for each iteration of the cross-validation
    # as well as the mean error rate versus lambda
    error_rate_visualization_ridge(lambdas, error_rate, filename)
