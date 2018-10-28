# -*- coding: utf-8 -*-

def get_lambda(model):
    """ Returns best regularization parameter for the model.
    
    Parameters
    ----------
    model: int
        The ID of the model. Possible values with meaning:
        0 - jet 0 with mass
        1 - jet 0 without mass
        2 - jet 1 with mass
        3 - jet 1 without mass
        4 - jet 2 with mass
        5 - jet 2 without mass
        6 - jet 3 with mass
        7 - jet 3 without mass
        
    Returns
    -------
    lambda_: float
        Best regularization parameter for the model.  
    """
    best_lambdas = [1000.0, 0.001, 100.0, 0.001, 100.0, 100.0, 0.001, 100.0]
    lambda_ = best_lambdas[model]
    return lambda_

def get_gamma(model):
    """ Returns best gamma parameter for the model.
    
    Parameters
    ----------
    model: int
        The ID of the model. Possible values with meaning:
        0 - jet 0 with mass
        1 - jet 0 without mass
        2 - jet 1 with mass
        3 - jet 1 without mass
        4 - jet 2 with mass
        5 - jet 2 without mass
        6 - jet 3 with mass
        7 - jet 3 without mass
        
    Returns
    -------
    gamma: float
        Best learning rate for the model.  
    """
    best_gammas = [1e-6, 1e-6, 1e-6, 1e-5, 1e-6, 1e-05, 1e-05, 1e-05]
    gamma = best_gammas[model]
    return gamma

def get_poly_degree(model):
    """ Returns best polynomial degree for the model.
    
    Parameters
    ----------
    model: int
        The ID of the model. Possible values with meaning:
        0 - jet 0 with mass
        1 - jet 0 without mass
        2 - jet 1 with mass
        3 - jet 1 without mass
        4 - jet 2 with mass
        5 - jet 2 without mass
        6 - jet 3 with mass
        7 - jet 3 without mass
        
    Returns
    -------
    degree: float
        Best polynomial degree for the model.
    """
    best_degrees = [3, 3, 3, 3, 3, 3, 3, 3]
    best_degree = best_degrees[model]
    return best_degree

def get_max_iters():
    """ Returns maximum number of iterations for gradient descent."""
    return 2000

def get_number_of_models():
    """ Returns number of models."""
    return 8