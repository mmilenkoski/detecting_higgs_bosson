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

def logistic_regression_sgd(y, tx, initial_w, max_iters, gamma):
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
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1):
            # update w and get loss
            loss, w = learning_by_gradient_descent(y_batch, tx_batch, w, gamma)
            if n_iter % 300 == 0:
                print("Itteration: %s, Loss: %s" % (n_iter, loss))
        
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent
    
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
    prev_loss = -1
    threshold = 1e-8
    
    for n_iter in range(max_iters):
        # update w and get loss
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        if n_iter % 300 == 0:
            print("Itteration: %s, Loss: %s" % (n_iter, loss))
        if prev_loss != -1 and np.abs(loss-prev_loss) < threshold:
            print (prev_loss)
            print(loss)
            break
        prev_loss = loss
        
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


def compute_loss(y, tx, w, method=None):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    if method == "logistic" or method == "SKL":
        return calculate_loss(y, tx, w)
    e = y - np.dot(tx, w)
    if method == "rmse":
        return np.sqrt(2* calculate_mse(e))
    #TUKA TREBA MAE ILI MSE???
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
            
def sigmoid(t):
    """apply sigmoid function on t."""
    """epsilon = 1e-5
    exponent = np.exp(t)
    print("exponent: %s" % np.isnan(exponent).any())
    return exponent/(1+exponent)"""
    """s = 1/(1+np.exp(-t))
    return s"""
    return np.exp(-np.logaddexp(0, -t))

    """def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x,dtype=float)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x,dtype=float)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)"""


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    """epsilon = 1e-5
    print("y: %s" % np.isnan(y).any())
    print("tx: %s" % np.isnan(tx).any())
    print("w: %s" % np.isnan(w).any())
    z = np.dot(tx, w)
    print("z: %s" % np.isnan(z).any())
    predicted = sigmoid(z)
    print("predicted: %s" % np.isnan(predicted).any())
    y.shape = (-1, 1)
    result = -np.sum(y * np.log(predicted + epsilon) + (1 - y) * np.log(1 - predicted + epsilon))
    #print(y)
    #print(predicted)
    return result/y.shape[0]"""
    """t = np.dot(tx, w)
    term1 = np.maximum(t, 0) + np.log(np.exp(-np.absolute(t)) + 1)
    #y.shape = (-1, 1)
    term2 = (np.multiply(y, t))
    erro = term1 - term2
    loss = sum(erro)
    print("Davor loss: %s" % (loss/y.shape[0]))"""
#     y.shape = (-1, 1)
#     pred = sigmoid(tx.dot(w))
#     loss = y.T.dot(np.log(pred + 1e-5)) + (1 - y).T.dot(np.log(1 - pred + 1e-5))
#     return np.squeeze(- loss)
    """compute the cost by negative log likelihood."""
    y.shape = (-1, 1)
    pred = tx.dot(w)
    term1 = np.logaddexp(0, pred)
    term2 = np.multiply(y, pred)
    loss = np.sum(term1-term2)
#     import sklearn
#     print("Real loss: %s" % (sklearn.metrics.log_loss(y, pred)))
#     print("Our loss: %s" % (loss/y.shape[0]))
    return loss/y.shape[0]

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    z = np.dot(tx, w)
    predicted = sigmoid(z)
    gradient = np.dot(tx.T, (predicted - y))
    return gradient

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    gradient = calculate_gradient(y, tx, w)
    w = w - gamma*gradient
    loss = calculate_loss(y, tx, w)
    return loss, w





