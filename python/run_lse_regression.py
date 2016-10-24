import numpy as np
from costs import compute_loss
from least_squares import least_squares
from feature_transformation import build_poly_matrix
from helpers import build_k_indices
from plots import cross_validation_visualization_lse


def cross_validation(y, x, k_indices, k):
    """return the loss of least squares."""
    # get k'th subgroup in test, others in train    
    indices_for_test = k_indices[k]
    x_test, y_test = x[indices_for_test], y[indices_for_test]
    x_training, y_training = np.delete(x, indices_for_test, axis=0), np.delete(y, indices_for_test, axis=0)

    # least_squares
    loss_tr, w_opt = least_squares(y_training, x_training)

    # calculate the loss for test data
    loss_te = compute_loss(y_test, x_test, w_opt)

    rmse_tr = np.sqrt(2 * loss_tr)
    rmse_te = np.sqrt(2 * loss_te)

    return rmse_tr, rmse_te


def run_least_squares(y, x, degrees=np.arange(0, 11), k_fold=4, seed=1, filename="cross_validation_lse"):
    """ Perform Least Squares using k-fold cross-validation and plot the training and test error.
        By default, the seed is 1 and the whole cross-validation process is done only once and the result is then plotted.
    """
    if k_fold <= 1:
        raise ValueError('The value of k_fold must be larger or equal to 2.')

    np.random.seed(seed)

    k_indices = build_k_indices(y, k_fold, seed)

    rmse_tr = np.zeros((k_fold, len(degrees)))
    rmse_te = np.zeros((k_fold, len(degrees)))

    # K-fold cross-validation of least_squares for every degree:
    for k in range(0, k_fold):
        for index_degree, degree in enumerate(degrees):
            phi = build_poly_matrix(np.copy(x), degree)
            rmse_tr[k, index_degree], rmse_te[k, index_degree] = cross_validation(y, phi, k_indices, k)

    # Plot the mean of training and test RMSE for every degree
    cross_validation_visualization_lse(degrees, np.mean(rmse_tr, axis=0), np.mean(rmse_te, axis=0), filename)
