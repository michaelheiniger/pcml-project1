# -*- coding: utf-8 -*-
""" Linear Regression using Stochastic Gradient Descent """

import numpy as np
from costs import compute_loss
from helpers import batch_iter


def least_squares_SGD(y, tx, gamma, max_iters):
    """Linear Regression using Stochastic Gradient Descent"""

    # initialization of the weights with zeros
    w_initial = np.array(np.zeros(tx.shape[1]))
    batch_size = 1
    # get the losses and weights for every iteration
    losses, ws = stochastic_gradient_descent(y, tx, w_initial, batch_size, max_iters, gamma)
    # take the best values
    w_opt = ws[max_iters - 1]
    loss = losses[max_iters - 1]

    return loss, w_opt


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    # Gradient of MSE loss (or cost) function is: 1/N transpose(X-tilde)*e

    e = y - tx.dot(w)
    gradient = -1 / y.shape[0] * np.transpose(tx).dot(e)

    # Value of the loss (or cost) function
    loss = compute_loss(y, tx, w)

    return loss, gradient


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    max_iters = max_epochs

    for n_iter in range(max_iters):
        for y_n, tx_n in batch_iter(y, tx, batch_size, True):
            loss, gradient = compute_stoch_gradient(y_n, tx_n, w)
            w = w - gamma * gradient

            # store w and loss
            ws.append(np.copy(w))
            losses.append(loss)

            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss,
                                                                                   w0=w[0], w1=w[1]))
    return losses, ws
