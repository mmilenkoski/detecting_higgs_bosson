# -*- coding: utf-8 -*-
import csv
import numpy as np
from utils.implementations import sigmoid


def load_csv_data(data_path, sub_sample=False):
    """ Loads data from csv.
    
    Parameters
    ----------
    data_path: string
        Data path of the data.
    sub_sample: boolean
        If True, returns subset of the data
        
    Returns
    -------
    yb: ndarray
        1D array containing the labels of the data.
    tX: ndarray
        2D array containing the data.
    ids: ndarray
        1D array containing the event ids.
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data, method):
    """Computes least squares using gradient descent.
    
    Parameters
    ----------
    weights: ndarray
        1D array containing the optimal weights.
    data: ndarray
        2D array containing the data.
    method: string
        If 'logistic', computes labels using sigmoid function.
        Otherwise, computes labels with dot product.
     
    Returns
    -------
    y_pred: ndarray
        1D array containing the predictions.
    """
    if len(weights.shape) == 1:
        weights.shape = (-1, 1)
    if method == "logistic":
        z = np.dot(data, weights)
        y_pred = sigmoid(z)
        y_pred[np.where(y_pred <= 0.5)] = -1
        y_pred[np.where(y_pred > 0.5)] = 1
    else:
        y_pred = np.dot(data, weights)
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """ Creates an output file in csv format for submission to Kaggle.
    
    Parameters
    ----------
    ids: ndarray
        Event ids.
    y_pred: ndarray
        1D array containing the predictions.
    name: string
        Data path of the output file to be created.
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
def accuracy_score(y_pred, y_true):
    """ Computes classification accuracy.
    
    Parameters
    ----------
    y_pred: ndarray
        1D array containing the predicted labels.
    y_true: ndarray
        1D array containing the true labels.
    
    Returns
    -------
    accuracy: float
        The computed classification accuracy.
    """
    assert y_pred.shape == y_true.shape
    accuracy =  np.sum(y_pred == y_true)*1.0/len(y_true)
    return accuracy

def transform_labels_to_zero_one(labels):
    """ Transforms the input labels in format for logistic regresion:
    
    Example: 
        -1 -> 0
        1 -> 1
    
    Parameters
    ----------
    labels: ndarray
        1D array containing the labels to be transformed.
    
    Returns
    -------
    transformed: ndarray
        Transofrmed labels.
    """
    transformed = (1 + labels) / 2
    return transformed