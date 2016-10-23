import numpy as np

def add_cos_function(tx):
    """ Augment the matrix tx with the cos of all its features: new_tx = [tx, cos(tx)] """
    
    copy_tx = np.copy(tx[:,1:]) # don't copy the first column of 1's
    cos_tx = np.cos(copy_tx)
    augmented_tx = np.concatenate((tx, cos_tx), axis=1)
    
    return augmented_tx   

def add_exp_function(tx):
    """ Augment the matrix tx with the exp of all its features: new_tx = [tx, exp(tx)] """
    
    copy_tx = np.copy(tx[:,1:]) # don't copy the first column of 1's
    exp_tx = np.exp(copy_tx)
    augmented_tx = np.concatenate((tx, exp_tx), axis=1)
    
    return augmented_tx   

