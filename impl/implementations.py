from impl.costs import *
from impl.gradient_descent import *
from impl.helpers import *
import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Computes least squares using gradient descent
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    initial_w: ndarray
        1D array containing the initial weight vector
    max_iters: int
        number of steps to run the gradient descent
    gamma: float
        step size
     
    Returns
    -------
    w: ndarray
        1D array containing the final weight vector
    loss: float
        loss corresponding to the last weight vector
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = -1
    
    for n_iter in range(max_iters):
        # compute gradient
        gradient, _ = compute_gradient(y, tx, w)
        # calculate loss
        loss = compute_loss(y, tx, w)
        # update w
        w = w - gamma * gradient
        
    return w, loss

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Computes least squares using stochastic gradient descent
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    initial_w: ndarray
        1D array containing the initial weight vector
    batch_size: int
        number of training examples used in one iteration of stochastic
        gradient descent
    max_iters: int
        number of steps to run the gradient descent
    gamma: float
        step size
     
    Returns
    -------
    w: ndarray
        1D array containing the final weight vector
    loss: float
        loss corresponding to the last weight vector
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = -1
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient
            gradient, _ = compute_gradient(y_batch, tx_batch, w)
            # calculate loss
            loss = compute_loss(y, tx, w)
            # update w
            w = w - gamma * gradient
            
    return w, loss

def least_squares(y, tx):
    """ Computes least squares using normal equations.
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
        
    Returns
    -------
    w: ndarray
        1D array containing the optimal weight vector
    loss: float
        loss corresponding to the optimal weight vector
    """
    # calculate optimal weights
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    # calculate loss
    loss = compute_loss(y, tx, w)
    
    return w, loss

def ridge_regression(y, tx, lambda_):
    """ Implements ridge regression.
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    lambda_: float
        Regularization parameter
    
    Returns
    -------
    w: ndarray
        1D array containing the optimal weight vector
    loss: float
        loss corresponding to the optimal weight vector   
    """
    # number of training examples
    N = tx.shape[0]
    # number of features
    D = tx.shape[1]
    # calculate optimal weights
    a = tx.T.dot(tx) + 2 * N * lambda_ * np.identity(D)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    # calculate loss
    loss = compute_loss(y, tx, w)
    
    return w, loss




