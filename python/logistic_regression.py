import numpy as np
from helpers import batch_iter, sigmoid


def logistic_regression(y, tx, gamma,  max_iters):
    """ Logistic regression using Stochastic gradient descent algorithm. Returns the optimal loss and w"""
    
    # Define parameters to store w and loss
    w = np.zeros((tx.shape[1], 1))
    losses = []
    #define batch_size
    batch_size= 100

    np.seterr(all='print')
    
    for iter in range(max_iters):
        
        y_batch, tx_batch = next(batch_iter(y, tx, batch_size, num_batches=1, shuffle=True))
        
        grad, loss = compute_gradient_log_likelihood(y_batch, tx_batch, w), compute_log_likelihood(y_batch, tx_batch, w)
        
        w = np.subtract(w, np.multiply(gamma, grad))
        
        losses.append(loss)
        
        if iter % 10 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break  
        
    return losses[len(losses)-1], w


def compute_log_likelihood(y, tx, w):
    """computes the loss (negative log likelihood) for logistic regression"""
    loss = 0
    for i in range(tx.shape[0]):
        y_est = tx[i].dot(w)
        
        # prevent overflow by assuming log(1+exp(k)) = k ,for large k
        if y_est > 100: 
            loss += y_est - y[i] * y_est
        else:
            loss +=  np.log(1 + np.exp(y_est)) - y[i] * y_est
    return loss


def compute_gradient_log_likelihood(y, tx, w):
    """Computes gradient of the max likelihood estimator for logistic regression"""
    
    xt_t = tx.transpose()
    return xt_t.dot((sigmoid(tx.dot(w)) - y))



