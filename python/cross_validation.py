from costs import compute_loss
from build_polynomial import build_poly
from regression_functions import *


def cross_validation(y, x, k_indices, k, lambda_, degree, max_iters, gamma, method):
    """return the training and test error obtained by applying the method passed as a parameter."""

    # get k'th subgroup in test, others in train:
    x_test = np.copy(x[k_indices[k]]).ravel()
    x_train = np.copy(x[np.delete(k_indices, k, 0)]).ravel()
    y_test = np.copy(y[k_indices[k]]).ravel()
    y_train = np.copy(y[np.delete(k_indices, k, 0)]).ravel()

    # form data with polynomial degree:
    phi_train = build_poly(x_train, degree)
    phi_test = build_poly(x_test, degree)

    weights_train = np.zeros(phi_train.shape[1])
    loss_train = 0

    # get the optimal weights acoording to the method passed as parameter:
    if method == 'least_squares_GD':
        weights_train, loss_train = least_squares_GD(y_train, phi_train, gamma, max_iters)
    elif method == 'least_squares_SGD':
        weights_train, loss_train = least_squares_SGD(y_train, phi_train, gamma, max_iters)
    elif method == 'least_squares':
        weights_train, loss_train = least_squares(y_train, phi_train)
    elif method == 'ridge_regression':
        weights_train, loss_train = ridge_regression(y_train, phi_train, lambda_)
    elif method == 'logistic_regression':
        weights_train, loss_train = logistic_regression(y_train, phi_train, gamma, max_iters)
    elif method == 'reg_logistic_regression':
        weights_train, loss_train = reg_logistic_regression(y_train, phi_train, lambda_, gamma, max_iters)

    # calculate the loss for train and test data:
    loss_tr = np.sqrt(2 * loss_train)
    loss_te = np.sqrt(compute_loss(y_test, phi_test, weights_train))

    return loss_tr, loss_te


def cross_validation_demo(y, x):
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 2, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mse_tr = []
    mse_te = []
    # ***************************************************
    # cross validation:
    for lambda_ in lambdas:
        # accumulate the losses for the k_fold repetitions, then average
        loss_tr_acc = 0
        loss_te_acc = 0
        for k in range(0, k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, lambda_, degree)
            loss_tr_acc += loss_tr
            loss_te_acc += loss_te
        mse_tr.append(loss_tr_acc / k_fold)
        mse_te.append(loss_te_acc / k_fold)

    return mse_te, mse_tr


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
