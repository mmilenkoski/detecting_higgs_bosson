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
        1D array containing the initial weight vector.
    max_iters: int
        Maximum number of steps to run the gradient descent.
    gamma: float
        Learning rate for gradient descent.
     
    Returns
    -------
    w: ndarray
        1D array containing the final weight vector.
    loss: float
        Loss corresponding to the last weight vector.
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = -1
    
    for n_iter in range(max_iters):
        # compute gradient
        gradient, e = compute_gradient_least_sqaures(y, tx, w)
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
        1D array containing the initial weight vector.
    batch_size: int
        Number of training examples used in one iteration of stochastic gradient descent.
    max_iters: int
        Maximum number of steps to run the gradient descent.
    gamma: float
        Learning rate for gradient descent.
     
    Returns
    -------
    w: ndarray
        1D array containing the final weight vector.
    loss: float
        Loss corresponding to the last weight vector.
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = -1
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size):
            # compute a stochastic gradient
            gradient, e = compute_gradient_least_sqaures(y_batch, tx_batch, w)
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
        1D array containing the optimal weight vector.
    loss: float
        Loss corresponding to the optimal weight vector.
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
        Regularization parameter.
    
    Returns
    -------
    w: ndarray
        1D array containing the optimal weight vector.
    loss: float
        Loss corresponding to the optimal weight vector.
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

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Implements logistic regression using gradient descent
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    initial_w: ndarray
        1D array containing the initial weight vector.
    max_iters: int
        Maximum number of steps to run the gradient descent.
    gamma: float
        Learning rate.
     
    Returns
    -------
    w: ndarray
        1D array containing the final weight vector.
    loss: float
        Loss corresponding to the last weight vector.
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = -1
    prev_loss = -1
    # Define threshold for early stopping of gradient descent
    threshold = 1e-8
    
    for n_iter in range(max_iters):
        # update w and calculate loss
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # stop the gradient descent if the difference between the last two losses is below the threshold
        if prev_loss != -1 and np.abs(loss-prev_loss) < threshold:
            break
        prev_loss = loss
        
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Implements regularized logistic regression using gradient descent
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    lambda_: float
        Regularization parameter.
    initial_w: ndarray
        1D array containing the initial weight vector.
    max_iters: int
        Maximum number of steps to run the gradient descent.
    gamma: float
        Learning rate.
     
    Returns
    -------
    w: ndarray
        1D array containing the final weight vector.
    loss: float
        Loss corresponding to the last weight vector.
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = -1
    prev_loss = -1
    # Define threshold for early stopping of gradient descent
    threshold = 1e-8
    
    for n_iter in range(max_iters):
        # update w and calculate loss
        loss, w = learning_by_reg_gradient_descent(y, tx, w, gamma, lambda_)
        # stop the gradient descent if the difference between the last two losses is below the threshold
        if prev_loss != -1 and np.abs(loss-prev_loss) < threshold:
            break
        prev_loss = loss
        
    return w, loss


def compute_gradient_least_sqaures(y, tx, w):
    """Computes the gradient for least squares.
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    w: ndarray
        1D array containing the weight vector.
     
    Returns
    -------
    grad: ndarray
        1D array containing the gradient.
    err: ndarray
        1D array containing the error for each training example.
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def calculate_mse(e):
    """Calculates the mean square error.
    
    Parameters
    ----------
    e: ndarray
        1D array representing the error vector.
     
    Returns
    -------
    mse: float
        Mean squared error.
    """
    n = e.shape[0]
    mse = (1/2)*np.sum(e**2)/n
    return mse


def calculate_mae(e):
    """Calculates the mean absolute error.
    
    Parameters
    ----------
    e: ndarray
        1D array representing the error vector.
     
    Returns
    -------
    mae: float
        Mean absolute error.
    """
    n = e.shape[0]
    mae = np.sum(np.abs(e))/n
    return mae


def compute_loss(y, tx, w, method=None):
    """Calculates the loss using negative log likelihood, rmse, mae or mse.

    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    w: ndarray
        1D array containing the weight vector
    method: string, optional
        Method for calculating the loss. 
        If 'logistic', calculate Negative Log-Likelihood.
        If 'rmse', calculates Root Mean Squared Error.
        If 'mae', calculates Mean Absolute Error.
        If None (default), calculates Mean Squared Error.
        
    Returns
    -------
    loss: float64
        The corresponding loss computed on the input.
    """
    e = y - np.dot(tx, w)
    
    # return the correct loss based on the value of `method` or mse by default.
    if method == "logistic":
        return negative_log_likelihood(y, tx, w)
    
    if method == "rmse":
        return np.sqrt(2*calculate_mse(e))
    
    if method == "mae":
        return calculate_mae(e)
    
    return calculate_mse(e)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generates a minibatch iterator for a dataset.
    
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    batch_size: int
        Integer representing the size of the batch returned by the iterator.
    num_batches: int
        Integer representing the number of batches to be returned.
    shuffle: bool
        Boolean value indicating if the data should be shuffled when returned as mini batches.
        
    Yields
    ------
    tuple
         Tuple containing the next batch of training examples for gradient descent in the form (labels, training_data).
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
    """Implements sigmoid function.
    
    Parameters
    ----------
    y: ndarray
        Input for the sigmoid function.
     
    Returns
    -------
    result: ndarray
        Transformed values using the sigmoid function.
    """
    result = np.exp(-np.logaddexp(0, -t))
    return result


def negative_log_likelihood(y, tx, w, lambda_=None):
    """Calculates the mean negative log-likelihood loss.
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    w: ndarray
        1D array containing the weight vector.
    lambda_: float, optional
        Regularization parameter for regularized logistic regression. 
        If None (default), computes loss for basic logistic regression.
     
    Returns
    -------
    loss: float
        Mean negative log-likelihood loss.
    """
    # calculate basic loss
    y.shape = (-1, 1)
    pred = tx.dot(w)
    # calculates log[1 + e^(pred)]
    term1 = np.logaddexp(0, pred)
    # element-wise multiplication
    term2 = np.multiply(y, pred)
    loss = np.sum(term1-term2)
    
    # add regularization term
    if lambda_ is not None:
        loss = loss + lambda_ * np.squeeze(np.dot(w.T, w))

    return loss/y.shape[0]

def compute_gradient_log_reg(y, tx, w, lambda_=None):
    """Computes the gradient for (regularized) logistic regression.
    
    If the regularization parameter is None, the function computes the gradient of logistic regression. Otherwise, it computes 
    the gradient of regularized logistic regression, using the value of lambda_ as regularization parameter.
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    w: ndarray
        1D array containing the weight vector.
    lambda_: float, optional
        Regularization parameter for regularized logistic regression. 
        If None (default), computes gradient for basic logistic regression.
    
    Returns
    -------
    gradient: ndarray
        1D array containing the gradient
    """
    # calculate basic gradient
    z = np.dot(tx, w)
    predicted = sigmoid(z)
    gradient = np.dot(tx.T, (predicted - y))
    
    # add regularization term
    if lambda_ is not None:
        gradient = gradient + 2*lambda_*w
    return gradient

def learning_by_gradient_descent(y, tx, w, gamma):
    """Implements one step of gradient descent for logistic regression and calculates the loss.
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    w: ndarray
        1D array containing the weight vector.
        
    Returns
    -------
    w: ndarray
        1D array containing the updated weight vector.
    loss: float
        Loss corresponding to the updated weight vector.
    """
    gradient = compute_gradient_log_reg(y, tx, w)
    w = w - gamma*gradient
    loss = negative_log_likelihood(y, tx, w)
    return loss, w

def learning_by_reg_gradient_descent(y, tx, w, gamma, lambda_):
    """Implements one step of gradient descent for regularized logistic regression and calculates the loss.
    
    Parameters
    ----------
    y: ndarray
        1D array containing the correct labels of the training data. 
    tx: ndarray
        2D array containing the training data.
    w: ndarray
        1D array containing the weight vector.
    lambda_: float
        Regularization parameter.
        
    Returns
    -------
    w: ndarray
        1D array containing the updated weight vector.
    loss: float
        Loss corresponding to the updated weight vector.
    """
    gradient = compute_gradient_log_reg(y, tx, w, lambda_)
    w = w - gamma*gradient
    loss = negative_log_likelihood(y, tx, w, lambda_)
    return loss, w