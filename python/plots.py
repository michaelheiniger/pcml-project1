# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt
   
def rmse_vs_lambda_visualization_ridge(lambdas, rmse_tr, rmse_te, filename):
    """visualize the bias variance decomposition."""
    
    # Compute the mean RMSE over each lambda for the training and test sets
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    
    plt.plot(
        lambdas,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        linewidth=0.3)
    plt.plot(
        lambdas,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        linewidth=0.3)
    plt.plot(
        lambdas,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='mean train',
        linewidth=3)
    plt.plot(
        lambdas,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='mean test',
        linewidth=3)
    plt.xlabel("lambdas")
    plt.ylabel("RMSE")
    plt.xscale('log')
    plt.legend(loc=2)
    plt.grid(True)
    plt.title("RMSE vs lambda")
    plt.savefig(filename)
    plt.clf() # needed in case of consecutive call of this function to avoid stacking unrelated plots 
    
    
def error_rate_visualization_ridge(lambdas, error_rate, filename):
    """ Visualize error rate versus lambda:
	The goal is to assess the model based on the error rate obtained for different lambdas and  		The error rate is obtained using cross-validation: we use a dataset containing known 		predictions and split it (e.g. k-fold) in a training and test dataset. We learn the model on 		the training set and use it on the test set. Then, we compare the predictions of the model 		for the test set with the actual values to compute the error rate.
	This function plots the mean error rate for each lambda as well as the error rate of each 		iterations so that we can see if there is variance or not and for what values of lambda. """
    
    # Compute the mean error rate over each lambda
    mean_error_rate = np.expand_dims(np.mean(error_rate, axis=0), axis=0)
    print(print("Minimum mean:", np.amin(mean_error_rate), ", lambda:", np.argmin(mean_error_rate)))

    plt.plot(
        lambdas,
        error_rate.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        linewidth=0.3)
    
    plt.plot(
        lambdas,
        mean_error_rate.T,
        'b',
        linestyle="-",
        label='mean error rate',
        linewidth=3)
    plt.xlabel("lambdas")
    plt.ylabel("Error rate")
    plt.xscale('log')
    plt.legend(loc=2)
    plt.grid(True)
    plt.title("Error rate versus lambda")
    plt.savefig(filename)
    plt.clf() # needed in case of consecutive call of this function to avoid stacking unrelated plots 
    
    
    

