import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.zeros((x.shape[0], degree + 1))
    for d in range(0, degree + 1):
        phi[:, d] = np.power(x, d)
    return phi


def build_poly_matrix(tx, degree):
    """polynomial basis function (without cross-terms) for input data tx, where the first column of tx is the
        all-one vector. The matrix x is then the matrix without the first column and the returned matrix is of
        the form [1 x^1 ... x^(degree)]"""
    
    dummy = np.copy(tx[:, 1:])  # don't copy the first column of 1's
    augmented_tx = np.copy(tx)
    
    # concatenate the powers of 'dummy' with the original matrix
    if degree >=2:
        for d in range(2,degree+1):
            augmented_tx = np.concatenate((augmented_tx, np.power(dummy, d)), axis =1)

    return augmented_tx



def add_cos_function(tx):
    """ Augment the matrix tx with the cos of all its features: new_tx = [tx, cos(tx)] """

    copy_tx = np.copy(tx[:, 1:])  # don't copy the first column of 1's
    cos_tx = np.cos(copy_tx)
    augmented_tx = np.concatenate((tx, cos_tx), axis=1)

    return augmented_tx


def add_exp_function(tx):
    """ Augment the matrix tx with the exp of all its features: new_tx = [tx, exp(tx)] """

    copy_tx = np.copy(tx[:, 1:])  # don't copy the first column of 1's
    exp_tx = np.exp(copy_tx)
    augmented_tx = np.concatenate((tx, exp_tx), axis=1)

    return augmented_tx


def add_sinc_function(tx):
    """ Augment the matrix tx with the sinc of all its features: new_tx = [tx, sinc(tx)] """

    copy_tx = np.copy(tx[:, 1:])  # don't copy the first column of 1's
    sinc_tx = np.sinc(copy_tx)
    augmented_tx = np.concatenate((tx, sinc_tx), axis=1)

    return augmented_tx


def add_functions(tx):
    
    copy_tx = np.copy(tx[:,1:])
    exp_tx = np.exp(copy_tx)
    cos_tx = np.cos(copy_tx)
    sin_tx = np.sin(copy_tx)
    sinc_tx = np.sinc(copy_tx)
    augmented_tx = np.concatenate((tx, exp_tx,), axis=1)
    return augmented_tx





