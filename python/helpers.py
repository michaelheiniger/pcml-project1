# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import math

def sigmoid(t):
    result = np.empty(np.shape(t))
    id_pos = t > 100
    id_neg = t < -100
    id_rest = np.logical_and(-100 <= t , t <= 100)
    #prevent over/underflow by assigning the limits of the sigmoid for large values of t
    result[id_pos] = 1.
    result[id_neg] = 0.
    result[id_rest] = np.divide(1, np.add(1, np.exp(-t)))
    return result

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold cross-validation."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # Set seed
    np.random.seed(seed)

    # Get size of the training set
    count_training = math.ceil(len(y) * ratio)

    # Get indices from 0 to N in randomized order
    indices = np.random.permutation(len(y))

    # The first count_training elements of indices are for training set
    # The remaining is for test set
    training_indice, test_indice = indices[:count_training], indices[count_training:]

    # Use the splitted indices to split the data for both x and y
    x_training, x_test = x[training_indice], x[test_indice]
    y_training, y_test = y[training_indice], y[test_indice]

    return x_training, y_training, x_test, y_test


def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]

    tx = np.hstack((np.ones((x.shape[0], 1)), x))
    return tx, mean_x, std_x


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
