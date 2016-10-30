import numpy as np

#################################################################
# Least squares with gradient descent
#################################################################

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear Regression using Gradient Descent"""

    # initialization of the weights with zeros
    w_initial = np.zeros(tx.shape[1])
    # get the losses and weights for every iteration
    losses, ws = gradient_descent(y, tx, w_initial, max_iters, gamma)
    # take the best values
    w_opt = ws[max_iters - 1]
    loss = losses[max_iters - 1]

    return w_opt, loss

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
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

#################################################################
# Least squares with stochastic gradient descent
#################################################################
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear Regression using Stochastic Gradient Descent"""

    # initialization of the weights with zeros
    w_initial = np.array(np.zeros(tx.shape[1]))
    batch_size = 1
    # get the losses and weights for every iteration
    losses, ws = stochastic_gradient_descent(y, tx, w_initial, batch_size, max_iters, gamma)
    # take the best values
    w_opt = ws[max_iters - 1]
    loss = losses[max_iters - 1]

    return w_opt, loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    # Gradient of MSE loss (or cost) function is: 1/N transpose(X-tilde)*e

    e = y - tx.dot(w)
    gradient = -1 / y.shape[0] * np.transpose(tx).dot(e)

    # Value of the loss (or cost) function
    loss = compute_loss(y, tx, w)

    return loss, gradient

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_epochs, gamma):
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

            #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

#################################################################
# Least squares - normal equation
#################################################################
def least_squares(y, tx):
    """ Calculates the least squares solution """
    
    # Compute weights w by solving as Ax = b
    A = np.transpose(tx).dot(tx)
    b = np.transpose(tx).dot(y)
    w_opt = np.linalg.solve(A, b)

    loss = compute_loss(y, tx, w_opt)

    return w_opt, loss

#################################################################
# Ridge regression - normal equation
#################################################################
def ridge_regression(y, tx, lambda_):
    """ Implements ridge regression """

    # Compute weights w by solving Ax = b
    A = np.transpose(tx).dot(tx) + 2 * lambda_ * len(y) * np.identity(tx.shape[1])
    b = np.transpose(tx).dot(y)
    w_opt = np.linalg.solve(A, b)
    
    # Compute the loss
    loss = compute_loss_ridge(y, tx, w_opt)

    return w_opt, loss

#################################################################
# Logistic regression
#################################################################
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using Stochastic gradient descent algorithm. Returns the optimal loss and w"""
    
    # Define parameters to store w and loss
    w = np.zeros((tx.shape[1], 1))
    losses = []
    #define batch_size
    batch_size= 2000

    np.seterr(all='print')
    
    for iter in range(max_iters):
        
        y_batch, tx_batch = next(batch_iter(y, tx, batch_size, num_batches=1, shuffle=True))
        
        grad, loss = compute_gradient_log_likelihood(y_batch, tx_batch, w), compute_log_likelihood(y_batch, tx_batch, w)
        
        w = np.subtract(w, np.multiply(gamma, grad))
        losses.append(loss)
        
        if iter % 50 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break  
        
    return w, losses[len(losses)-1]


def compute_log_likelihood(y, tx, w):
    """
    Computes maximum log likelihood estimator for logistic regression.
    To avoid floating-point overflows we substitute ln(1+exp{x}) with x, for x large enough.
    To avoid underflows, we substitute ln(1 +exp{x}) with 0 for x small enough.

    """
    xw = tx.dot(w)
    p = xw > 50
    q = xw < -50
    not_p = np.logical_not(p)
    not_q = np.logical_not(q)
    not_qp = np.logical_and(not_p, not_q)
    L = np.zeros(xw.shape)
    L[p], L[q], L[not_qp] = xw[p], 0 ,np.log(1 + np.exp(xw[not_qp]))
    return np.sum(L - np.multiply(y, xw))

def compute_gradient_log_likelihood(y, tx, w):
    """Computes gradient of the max likelihood estimator for logistic regression"""
    xt_t = tx.transpose()
    return xt_t.dot((sigmoid(tx.dot(w)) - y))
    
#################################################################
# Regularized (or penalized) logistic regression
#################################################################
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ 
    Regularized Logistic regression using Stochastic gradient descent algorithm. 
    Returns the optimal loss and w
    """
    
    # Define parameters to store w and loss
    w = np.zeros((tx.shape[1], 1))
    losses = []
    
    #define batch_size
    batch_size= 2000

    np.seterr(all='print')
    
    for iter in range(max_iters):
        
        y_batch, tx_batch = next(batch_iter(y, tx, batch_size, num_batches=1, shuffle=True))
        
        grad, loss = compute_gradient_log_likelihood_penalized(y_batch, tx_batch, w, lambda_), compute_log_likelihood_penalized(y_batch, tx_batch, w, lambda_)
        
        w = np.subtract(w, np.multiply(gamma, grad))
        
        losses.append(loss)
        
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break  
        
    return w, losses[len(losses)-1]


def compute_log_likelihood_penalized(y, tx, w, lambda_):
    """ 
    Computes maximum log likelihood estimator for logistic regression.
    To avoid floating-point overflows we substitute ln(1+exp{x}) with x, for x large enough.
    To avoid underflows, we substitute ln(1 +exp{x}) with 0 for x small enough.
    """
    xw = tx.dot(w)
    p = xw > 50
    q = xw < -50
    not_p = np.logical_not(p)
    not_q = np.logical_not(q)
    not_qp = np.logical_and(not_p, not_q)
    L = np.zeros(xw.shape)
    L[p], L[q], L[not_qp] = xw[p], 0 ,np.log(1 + np.exp(xw[not_qp]))
    loss = np.sum(L - np.multiply(y, xw))
    # add the penalization term to the loss
    return np.add(loss, lambda_ * np.sum(w.transpose().dot(w)))


def compute_gradient_log_likelihood_penalized(y, tx, w, lambda_):
    """Computes gradient of the max likelihood estimator for logistic regression"""
    
    xt_t = tx.transpose()
    gradient = xt_t.dot((sigmoid(tx.dot(w)) - y))
    return np.add(gradient, np.multiply(2 * lambda_, w))
    

#################################################################
# Dependencies
#################################################################
def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e.dot(e))

def compute_loss(y, tx, w):
    """Calculate the loss.
        You can calculate the loss using mse or mae.
        """
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_loss_ridge(y, tx, w):
    """ Calculate the loss for ridge regression:
    The version cost function considered is: 1/(2*N) sum_over_n(y_n-x_n*w)^2 + lambda*norm(w)^2.
    """
    e = y - tx.dot(w)
    N = y.shape[0]
    return 1/(2*N)*(np.transpose(e).dot(e)).sum()

def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size / batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def sigmoid(t):
    result = np.empty(np.shape(t))
    id_pos = t > 100
    id_neg = t < -100
    id_rest = np.logical_and(-100 <= t , t <= 100)
    #prevent over/underflow by assigning the limits of the sigmoid for large values of t
    result[id_pos] = 1.
    result[id_neg] = 0.
    result[id_rest] = np.divide(1, np.add(1, np.exp(-t[id_rest])))
    return result
