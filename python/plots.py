# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
def cross_validation_visualization_lse(degrees, rmse_tr, rmse_te, filename):
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(degrees, rmse_tr, marker=".", color='b', label='train error')
    plt.semilogx(degrees, rmse_te, marker=".", color='r', label='test error')
    plt.xlabel("degree")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(filename)
    plt.clf() # needed in case of consecutive call of this function to avoid stacking unrelated plots 


def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    #print(rmse_tr_mean,"\n\n")
    #print(rmse_te_mean,"\n\n")
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")
    
def bias_variance_decomposition_visualization_ridge(lambdas, rmse_tr, rmse_te, filename):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    
    #rmse_tr_mean = np.divide(rmse_tr_mean, np.amax(rmse_tr_mean))
    #rmse_te_mean = np.divide(rmse_te_mean, np.amax(rmse_tr_mean))
    """
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
        linewidth=0.3)"""
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
    """visualize error rate versus lambda."""
    mean_error_rate = np.expand_dims(np.mean(error_rate, axis=0), axis=0)
    
    
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
    
    
    

