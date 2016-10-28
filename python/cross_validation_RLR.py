import numpy as np
import matplotlib.pyplot as plt
from helpers import build_k_indices
from implementations import reg_logistic_regression, compute_log_likelihood_penalized

def cross_validation_visualization(lambds, loss_tr, loss_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, loss_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, loss_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("neg log-likelihood")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    

def cross_validation(y, x, k_indices, k, lambda_, gamma, max_iters):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train    
    indices_for_test = k_indices[k]
    x_test, y_test = x[indices_for_test], y[indices_for_test]
    x_training, y_training = np.delete(x, indices_for_test, axis=0), np.delete(y, indices_for_test, axis=0)

    # ridge regression
    loss_tr, w_opt = reg_logistic_regression(y_training, x_training, lambda_, gamma, max_iters)

    # calculate the loss for test data
    loss_te = compute_log_likelihood_penalized(y_test, x_test, w_opt, lambda_)

    return loss_tr, loss_te


def run_reg_logistic_regression(y, x, gamma, max_iters, k_fold=4,lambdas=np.logspace(-3, 2, 20), 
                                seed=1, filename="bias_var_decom_RLR"):
    """ Perform Regularized Logistic regression using k-fold cross-validation and plot the training and test error. 
    By default, the seed is 1 and the whole cross-validation process is done only 
    once and the result is then plotted.
    """
    if k_fold <= 1:
        raise ValueError('The value of k_fold must be larger or equal to 2.')

    np.random.seed(seed)

    k_indices = build_k_indices(y, k_fold, seed)

    loss_tr = np.zeros((k_fold, len(lambdas)))
    loss_te = np.zeros((k_fold, len(lambdas)))

    # K-fold cross-validation:
    for k in range(0, k_fold):
        for index_lambda, lambda_ in enumerate(lambdas):
            loss_tr[k, index_lambda], loss_te[k, index_lambda] = cross_validation(y, x, k_indices, k, lambda_,
                                                                                 gamma, max_iters)

    # Plot the mean training and test loss for every lambda 
    cross_validation_visualization(lambdas, np.mean(loss_tr, axis =0), np.mean(loss_te, axis=0))


