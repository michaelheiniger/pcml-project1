import numpy as np
from costs import compute_loss
from ridge_regression import ridge_regression
from build_polynomial import build_poly
from build_k_indices import build_k_indices

def cross_validation(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train    
    indices_for_test = k_indices[k]
    x_test, y_test = x[indices_for_test], y[indices_for_test]
    x_training, y_training = np.delete(x, indices_for_test, axis=0), np.delete(y, indices_for_test, axis=0)
    
    # ridge regression
    loss_tr, w_opt = ridge_regression(y_training, x_training, lambda_)
    
    # calculate the loss for test data
    loss_te = compute_loss(y_test, x_test, w_opt)
    #print("Training: ", loss_tr, ", Test: ", loss_te)
    return loss_tr, loss_te


from plots import cross_validation_visualization

def run_ridge_regression(y, x):
    """ Ridge regression with K-fold cross-validation and plots of bias-variance decomposition """
    seed = 1
    k_fold = 6
    lambdas = np.logspace(-10, 10, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    losses_tr = np.zeros((k_fold, len(lambdas)))
    losses_te = np.zeros((k_fold, len(lambdas)))
                       
    # K-fold cross-validation:
    for k in range(0, k_fold):
        for l in range(0,len(lambdas)):
            losses_tr[k,l], losses_te[k,l] = cross_validation(y, x, k_indices, k, lambdas[l])
                          
    # Mean for each lambda: should the mean be computer on losses_tr or np.sqrt(2*losses_tr) ?
    rmse_tr = np.sqrt(2*np.mean(losses_tr, axis=0)) 
    rmse_te = np.sqrt(2*np.mean(losses_te, axis=0))
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)

from plots import bias_variance_decomposition_visualization_ridge
from split_data import split_data
    
def run_ridge_regression_bias_var(y, x):
    seeds = range(100)
    ratio_train = 0.1
    lambdas = np.logspace(-10, 10, 30)

    
    # define list to store the variable
    rmse_tr = np.empty((len(seeds), len(lambdas)))
    rmse_te = np.empty((len(seeds), len(lambdas)))
    
    for index_seed, seed in enumerate(seeds):
        np.random.seed(seed)
        # split data with a specific seed
        x_training, y_training, x_test, y_test = split_data(x, y, ratio_train, seed)
        
        # bias_variance_decomposition
        for index_lambda, lambda_ in enumerate(lambdas):
            loss_tr, w_opt = ridge_regression(y_training, x_training, lambda_)
            loss_te = compute_loss(y_test, x_test, w_opt)
            rmse_tr[index_seed, index_lambda] = np.sqrt(2*loss_tr)
            rmse_te[index_seed, index_lambda] = np.sqrt(2*loss_te)

    bias_variance_decomposition_visualization_ridge(lambdas, rmse_tr, rmse_te)
    
    
    
    
