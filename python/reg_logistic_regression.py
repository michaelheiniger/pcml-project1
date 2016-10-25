import numpy as np

def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    # init parameters
    # max_iter = 10000
    # gamma = 0.01
    # lambda_ = 0.1
    threshold = 1e-8
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))

        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return losses[len(losses)-1], w


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""

    loss = calculate_loss_reg_logistic_regression(y, tx, w, lambda_)
    grad = calculate_gradient_reg_logistic_regression(y, tx, w, lambda_)

    return loss, grad


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # return loss, gradient and hessian:
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    # update w:
    w -= gamma * grad
    return loss, w


def calculate_loss_reg_logistic_regression(y, tx, w, lambda_):
    """compute the cost by penalized negative log likelihood."""
    loss = 0
    for i in range(0, len(y)):
        k = tx[i, :].transpose().dot(w)
        # prevent overflow by assuming log(1+exp(k)) = k ,for large k
        if k > 100:
            loss += k - y[i] * k
        else:
            loss += np.log(1 + np.exp(k)) - y[i] * k


            # add the penalization term to the loss
    loss += lambda_ * np.sum(np.power(w,2))
    return loss


def calculate_gradient_reg_logistic_regression(y, tx, w, lambda_):
    """compute the gradient of loss. The gradient is given by X^T(sigma(Xw)-y)"""

    # initialization
    grad = np.empty(np.shape(w))
    M = tx.dot(w)
    sigma = np.empty(np.shape(M))

    # compute the sigmoid for every value in M
    for i in range(0, len(M)):
        sigma[i] = sigmoid(M[i])

    grad = np.transpose(tx).dot(sigma - y) + 2 * lambda_ * w
    return grad


def sigmoid(t):
    """apply sigmoid function on t."""
    result = 0

    if t > 100:
        result = 1.
    elif t < -100:
        result = 0.
    else:
        result = np.exp(t) / (1 + np.exp(t))
    return result
