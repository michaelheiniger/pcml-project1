
import numpy as np
from plots import bias_variance_decomposition_visualization
from split_data import split_data
from least_squares import least_squares
from costs import compute_loss
from build_polynomial import build_poly

def run_lse(y, x, degree):
    
    seed = 1
    ratio_train = 0.8
    x_training, y_training, x_test, y_test = split_data(x, y, ratio_train, seed)
    phi_training = build_poly(x_training, degree)
    phi_test = build_poly(x_test, degree)
    
    loss_tr, w_opt = least_squares(y_training, phi_training)
    loss_te = compute_loss(y_test, phi_test, w_opt)
    
    return loss_te, w_opt


def run_lse_regression_bias_var(y, x):
    seeds = range(100)
    ratio_train = 0.5
    degrees = range(1, 10)

    # define list to store the variable
    rmse_tr = np.empty((len(seeds), len(degrees)))
    rmse_te = np.empty((len(seeds), len(degrees)))
    
    for index_seed, seed in enumerate(seeds):
        np.random.seed(seed)
        # split data with a specific seed
        x_training, y_training, x_test, y_test = split_data(x, y, ratio_train, seed)
        
        # bias_variance_decomposition
        for index_degree, degree in enumerate(degrees):
            phi_training = build_poly(x_training, degree)
            phi_test = build_poly(x_test, degree)
            loss_tr, w_opt = least_squares(y_training, phi_training)
            loss_te = compute_loss(y_test, phi_test, w_opt)
            #print("Tr: ", loss_tr, "Te: ", loss_te)
            rmse_tr[index_seed, index_degree] = np.sqrt(2*loss_tr)
            rmse_te[index_seed, index_degree] = np.sqrt(2*loss_te)

    bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te)
    
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    rmse_te_min_index = np.argmin(rmse_te_mean)
    degree_min = degrees[rmse_te_min_index]
    print(rmse_te_mean.shape)
    print(rmse_te_min_index)
    rmse_te_min = rmse_te_mean[0,rmse_te_min_index]
    return rmse_te_min, degree_min
        