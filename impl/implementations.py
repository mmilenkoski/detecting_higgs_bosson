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
        gradient, e = compute_gradient(y, tx, w)
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
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size):
            # compute a stochastic gradient
            gradient, e = compute_gradient(y_batch, tx_batch, w)
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
    tx_transpose = np.transpose(tx)
    left = np.dot(tx_transpose, tx)
    right = np.dot(tx_transpose, y)
    w = np.linalg.solve(left, right)
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
    tx_transpose = np.transpose(tx)
    lambda_prim = 2 * N * lambda_
    left = np.dot(tx_transpose, tx) + lambda_prim  * np.eye(D)
    right = np.dot(tx_transpose, y)
    w = np.linalg.solve(left, right)
    # calculate loss
    loss = compute_loss(y, tx, w)
    
    return w, loss


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

def calculate_mse(e):
    """Calculate the mse for vector e."""
    n = e.shape[0]
    mse = (1/2)*np.sum(e**2)/n
    return mse


def calculate_mae(e):
    """Calculate the mae for vector e."""
    n = e.shape[0]
    mae = np.sum(np.abs(e))/n
    return mae


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - np.dot(tx, w)
    return calculate_mae(e)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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




