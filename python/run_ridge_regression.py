import numpy as np
from costs import compute_loss_ridge
from ridge_regression import ridge_regression
from build_polynomial import build_poly
from build_k_indices import build_k_indices
from plots import bias_variance_decomposition_visualization_ridge

def cross_validation(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train    
    indices_for_test = k_indices[k]
    x_test, y_test = x[indices_for_test], y[indices_for_test]
    x_training, y_training = np.delete(x, indices_for_test, axis=0), np.delete(y, indices_for_test, axis=0)
    
    # ridge regression
    loss_tr, w_opt = ridge_regression(y_training, x_training, lambda_)
    
    # calculate the loss for test data
    loss_te = compute_loss_ridge(y_test, x_test, w_opt)
    
    rmse_tr = np.sqrt(2*loss_tr)
    rmse_te = np.sqrt(2*loss_te)
    
    return rmse_tr, rmse_te

def run_ridge_regression(y, x, lambdas=np.logspace(-10, 10, 30), k_fold=10, seed=1, filename="bias_var_decom_ridge"):
    """ Perform Ridge regression using k-fold cross-validation and plot the training and test error. By default, the seed is 1 and the whole cross-validation process is done only once and the result is then plotted.
    """
    if k_fold <= 1:
        raise ValueError('The value of k_fold must be larger or equal to 2.')

    np.random.seed(seed)

    k_indices = build_k_indices(y, k_fold, seed)

    rmse_tr = np.zeros((k_fold, len(lambdas)))
    rmse_te = np.zeros((k_fold, len(lambdas)))

    # K-fold cross-validation:
    for k in range(0, k_fold):
        for index_lambda, lambda_ in enumerate(lambdas):
            rmse_tr[k, index_lambda], rmse_te[k, index_lambda] = cross_validation(y, x, k_indices, k, lambda_)


    # Plot the training and test RMSE for every (k, lambda) pair as-well as the mean over all k
    bias_variance_decomposition_visualization_ridge(lambdas, rmse_tr, rmse_te, filename)

