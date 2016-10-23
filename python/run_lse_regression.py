
import numpy as np
from costs import compute_loss
from least_squares import least_squares
from build_polynomial import build_poly_matrix
from build_k_indices import build_k_indices
from plots import bias_variance_decomposition_visualization_ridge

def cross_validation(y, x, k_indices, k):
    """return the loss of least squares."""
    # get k'th subgroup in test, others in train    
    indices_for_test = k_indices[k]
    x_test, y_test = x[indices_for_test], y[indices_for_test]
    x_training, y_training = np.delete(x, indices_for_test, axis=0), np.delete(y, indices_for_test, axis=0)
    
    # ridge regression
    loss_tr, w_opt = least_squares(y_training, x_training)
    
    # calculate the loss for test data
    loss_te = compute_loss(y_test, x_test, w_opt)
    
    return loss_tr, loss_te

def run_least_squares(y, x, degrees = np.arange(0,11), k_fold=4, seeds=np.array([1])):
    """ Perform Least Squares using k-fold cross-validation and plot the training and test error.
        By default, the seed is 1 and the whole cross-validation process is done only once and the result is then plotted.
        A list of seeds can be provided to do several (i.e. the number of seeds) times the whole cross-validation process.
        This would result in a bias-variance decomposition plot (see last part of lab 4)
        Note: A high number of seeds could induce a prohibitive computation time (around 10min for 100 seeds on my laptop) !
    """
    if k_fold <= 1:
        raise ValueError('The value of k_fold must be larger or equal to 2.')


    """ Store the mean of training and test RMSE for every (seed, degree) pair.
        The values are the output of the cross-validation process (see below)
    """
    rmse_tr = np.empty((len(seeds), len(degrees)))
    rmse_te = np.empty((len(seeds), len(degrees)))

    for index_seed, seed in enumerate(seeds):
        np.random.seed(seed)

        k_indices = build_k_indices(y, k_fold, seed)

        losses_tr = np.zeros((k_fold, len(degrees)))
        losses_te = np.zeros((k_fold, len(degrees)))

        # K-fold cross-validation of least_squares for every degree:
        for k in range(0, k_fold):
            for index_degree, degree in enumerate(degrees):
                phi = build_poly_matrix(x, degree)
                losses_tr[k, index_degree], losses_te[k, index_degree] = cross_validation(y, phi, k_indices, k)

        rmse_tr[index_seed] = np.mean(np.sqrt(2 * losses_tr), axis=0)
        rmse_te[index_seed] = np.mean(np.sqrt(2 * losses_te), axis=0)


    # Plot the mean of training and test RMSE for every (seed, degree) pair as-well as the mean over all seeds
    bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te)