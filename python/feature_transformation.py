import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.zeros((x.shape[0], degree + 1))
    for d in range(0, degree + 1):
        phi[:, d] = np.power(x, d)
    return phi


def build_poly_matrix(tx, degree):  # TODO: rewrite in the style of functions below?
    """polynomial basis function (without cross-terms) for input data x,
    the returned matrix is of the form [1 x^1 ... x^(degree)]"""
    rows = tx.shape[0]
    cols = int(tx.size / rows)
    dummy = np.copy(tx)  # don't copy the first column of 1's

    if cols == 1:  # add a second dimension to 1D array
        dummy = np.reshape(dummy, (rows, 1))

    # fill phi with the powers of tx
    phi = np.empty((rows, cols * degree + 1))

    # Add a column of one at the first position (offset)
    phi[:, 0] = np.ones((1, rows))
    for d in range(0, degree):
        phi[:, 1 + d * cols: 1 + (d + 1) * cols] = np.power(dummy, d + 1)
    return phi


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
