import numpy as np
from costs import compute_loss
from build_polynomial import build_poly
from regression_functions import *


def cross_validation_step(y, x, k_indices, k, degree, lambda_):
    """return the training and test error obtained by applying the method passed as a parameter."""

    # get k'th subgroup in test, others in train:
    test_indices = k_indices[k].ravel()
    train_indices = np.delete(k_indices, k, 0).ravel()
    x_test = np.copy(x[test_indices])
    x_train = np.copy(x[train_indices])
    y_test = np.copy(y[test_indices])
    y_train = np.copy(y[train_indices])

    # form data with polynomial degree:
    phi_train = build_poly(x_train, degree)
    phi_test = build_poly(x_test, degree)

    weights_train, loss_train = ridge_regression(y_train, phi_train, lambda_)

    # calculate the loss for train and test data:
    loss_tr = np.sqrt(2 * loss_train)
    loss_te = np.sqrt(2 * compute_loss(y_test, phi_test, weights_train))

    return loss_tr, loss_te


def cross_validation_kfold(y, x, degree, k_fold, lambdas):
    """Computes and returns the average test and train rmse for each lambda for polynomial fitting """

    seed = 1
    # lambdas = np.logspace(-4, 2, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # cross validation:
    for lambda_ in lambdas:
        # accumulate the losses for the k_fold repetitions, then average
        loss_tr_acc = 0
        loss_te_acc = 0
        for k in range(0, k_fold):
            loss_tr, loss_te = cross_validation_step(y, x, k_indices, k, degree, lambda_)
            loss_tr_acc += loss_tr
            loss_te_acc += loss_te
        rmse_tr.append(loss_tr_acc / k_fold)
        rmse_te.append(loss_te_acc / k_fold)

    return rmse_te, rmse_tr


def get_best_lambda(y, x, degree, k_fold, lambdas):
    """determines the best lambda for a polynomial fitting using kfold cross validation"""
    rmse_te, rmse_tr = cross_validation_kfold(y, x, degree, k_fold, lambdas)
    min_idx = rmse_te == np.min(rmse_te)

    return lambdas[min_idx], np.min(rmse_te)


def get_best_degree(y, x, degrees, k_fold, lambdas):
    """determines the best degree for polynomial fitting and returns it alonf with the corresponding lambda and the
     achieved test error"""

    best_lambdas = np.zeros(degrees.shape)
    min_errors = np.zeros(degrees.shape)

    # for every degree we obtain the best lambda and its corresponding test error
    for i in range(0, len(degrees)):
        lamb, err = get_best_lambda(y, x, degrees[i], k_fold, lambdas)
        best_lambdas[i] = lamb
        min_errors[i] = err

    # bla = np.zeros((3, len(degrees)))
    # bla[0, :] = degrees
    # bla[1, :] = best_lambdas
    # bla[2, :] = min_errors

    mask = min_errors == np.min(min_errors)
    return degrees[mask], best_lambdas[mask], np.min(min_errors)


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
