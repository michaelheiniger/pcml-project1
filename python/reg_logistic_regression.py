import numpy as np
from helpers import batch_iter, sigmoid

def reg_logistic_regression(y, tx, lambda_, gamma,  max_iters):
    """ Regularized Logistic regression using Stochastic gradient descent algorithm. Returns the optimal loss and w"""
    
    # Define parameters to store w and loss
    w = np.zeros((tx.shape[1], 1))
    losses = []
    
    #define batch_size
    batch_size= 100

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
        
    return losses[len(losses)-1], w


def compute_log_likelihood_penalized(y, tx, w, lambda_):
    """computes the loss (negative log likelihood+ penalization term) for regularized logistic regression"""
    
    loss = 0
    for i in range(tx.shape[0]):
        y_est = tx[i].dot(w)
        # prevent overflow by assuming log(1+exp(k)) = k ,for large k
        if y_est > 100: 
            loss += y_est - y[i] * y_est
        else:
            loss +=  np.log(1 + np.exp(y_est)) - y[i] * y_est
    # add the penalization term to the loss
    return np.add(loss, lambda_ * w.transpose().dot(w))


def compute_gradient_log_likelihood_penalized(y, tx, w, lambda_):
    """Computes gradient of the max likelihood estimator for logistic regression"""
    
    xt_t = tx.transpose()
    gradient = xt_t.dot((sigmoid(tx.dot(w)) - y))
    return np.add(gradient, np.multiply(2 * lambda_, w))
