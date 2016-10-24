# -*- coding: utf-8 -*-
""" Linear Regression using Gradient Descent """

import numpy as np
from costs import compute_loss


def least_squares_GD(y, tx, gamma, max_iters):
    """ Linear Regression using Gradient Descent"""

    # initialization of the weights with zeros
    w_initial = np.zeros(tx.shape[1])
    # get the losses and weights for every iteration
    losses, ws = gradient_descent(y, tx, w_initial, max_iters, gamma)
    # take the best values
    w_opt = ws[max_iters - 1]
    loss = losses[max_iters - 1]

    return loss, w_opt


def compute_gradient(y, tx, w):
    """Compute the gradient of the MSE loss function."""

    # Gradient of MSE loss (or cost) function is: 1/N transpose(X-tilde)*e
    e = y - tx.dot(w)
    gradient = -1 / y.shape[0] * np.transpose(tx).dot(e)

    # Value of the MSE loss (or cost) function
    loss = compute_loss(y, tx, w)
    return loss, gradient


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss for every iteration
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss, gradient = compute_gradient(y, tx, w)

        # Update rule for Gradient Descent is
        # w(t+1) = w(t) - gamma * gradient(w(t))
        w = w - gamma * gradient

        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
