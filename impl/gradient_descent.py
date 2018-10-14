# -*- coding: utf-8 -*-
from impl.costs import calculate_mse


def compute_gradient(y, tx, w):
    """Computes the gradient.
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    w: ndarray
        1D array containing the weight vector
     
    Returns
    -------
    grad: ndarray
        1D array containing the gradient
    err: ndarray
        1D array containing the errors for each training point
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err