# -*- coding: utf-8 -*-
"""user defined function for cross validation."""
import numpy as np
from impl.implementations import *
from utils.helpers import *
from utils.preprocessing import *
from utils.plots import *


def cross_validation(x, y, lambdas, poly_degree=-1, norm=None, method="ridge", n_splits=10, visualize=True, seed=0, max_iters=100):
    print ("Start")
    rmse_tr = []
    rmse_te = []
    acc_tr = []
    acc_te = []
    if method == "logistic" or method == "SKL":
        y = (1 + y) / 2
    
    for t, lambda_ in enumerate(lambdas):
        print("NEW LAMBDA: %s" % lambda_)
        rmse_tr_t = []
        rmse_te_t = []
        acc_tr_t = []
        acc_te_t = []
        for train_ind, test_ind in k_fold_split(y=y, x=x, n_splits=n_splits, seed=seed):
            x_train_kfold = x[train_ind].copy()
            y_train_kfold = y[train_ind]
            x_test_kfold = x[test_ind].copy()
            y_test_kfold = y[test_ind]
            
            if norm == "min_max":
                x_train_kfold, x_test_kfold = min_max_normalization(x_train_kfold, x_test_kfold)
            elif norm == "std":
                x_train_kfold, x_test_kfold = standardize(x_train_kfold, x_test_kfold)
                
            x_train_kfold = build_poly(x_train_kfold, poly_degree)
            x_test_kfold = build_poly(x_test_kfold, poly_degree)
            
            
            initial_w = np.random.randn(x_train_kfold.shape[1])
            initial_w.shape = (-1, 1)
            if method == "ridge":
                w, _ = ridge_regression(tx=x_train_kfold, y=y_train_kfold, lambda_=lambda_)
            elif method == "logistic":
                y_train_kfold.shape = (-1, 1)
                w, _ = logistic_regression(y=y_train_kfold, tx=x_train_kfold, initial_w=initial_w, max_iters=max_iters, gamma=lambda_)
            elif method =="SKL":
                from sklearn.linear_model import LogisticRegression as SLR
                slr = SLR()
                y_train_kfold.shape = (-1)
                slr = slr.fit(x_train_kfold, y_train_kfold)
                w = slr.coef_
                w.shape = (-1, 1)
            #elif: TO ADD NEW METHODS
            
            if method == "logistic" or method == "SKL":
                loss_method = "logistic"
            else:
                loss_method = "rmse"
            
            loss_tr = compute_loss(y_train_kfold, x_train_kfold, w, loss_method)
            loss_te = compute_loss(y_test_kfold, x_test_kfold, w, loss_method)
            rmse_tr_t.append(loss_tr)
            rmse_te_t.append(loss_te)
            
            train_pred = predict_labels(w, x_train_kfold, method)
            test_pred = predict_labels(w, x_test_kfold, method)
            
            if method == "logistic" or method == "SKL":
                y_train_kfold = 2*y_train_kfold - 1
                y_test_kfold = 2*y_test_kfold - 1
            acc_tr_t.append(np.sum(y_train_kfold == train_pred)*1.0/len(train_pred))
            acc_te_t.append(np.sum(y_test_kfold == test_pred)*1.0/len(test_pred))
            
            
            
            """from sklearn.metrics import log_loss
            z = np.dot(x_train_kfold, w)
            train_pred = sigmoid(z)
            z = np.dot(x_test_kfold, w)
            test_pred = sigmoid(z)
            print("Training loss: %s, Testing loss: %s" % (loss_tr, loss_te))
            print("SCTraining loss: %s, SCTesting loss: %s" % (log_loss(y_train_kfold, train_pred), log_loss(y_test_kfold, test_pred)))"""
            
            
        rmse_tr.append(np.mean(rmse_tr_t))
        rmse_te.append(np.mean(rmse_te_t))
        acc_tr.append(np.mean(acc_tr_t))
        acc_te.append(np.mean(acc_te_t))
    print (visualize)
    if visualize:
        cross_validation_visualization(lambdas, rmse_tr, rmse_te)
        print("Train loss: %s" % (rmse_tr))
        print("Test loss: %s" % (rmse_te))
        #print("Train acc: %s" % (np.mean(acc_tr)))
        #print("Test acc: %s" % (np.mean(acc_te)))
        print()
    lambda_ind = np.argmax(acc_te)
    return lambdas[lambda_ind], acc_tr[lambda_ind], acc_te[lambda_ind]

def train_and_get_predictions(X_train, Y_train, X_test, Y_test, Y_inds, best_lambdas, poly_degree=-1, norm=None, method="ridge", seed=0):
    predictions = np.ones(568238)
    train_accuracy = []
    for i, lambda_ in enumerate(best_lambdas):
        x_train = X_train[i].copy()
        y_train = Y_train[i]
        x_test = X_test[i].copy()
        y_test = Y_test[i]
        
        if norm == "min_max":
            x_train, x_test = min_max_normalization(x_train, x_test)
        elif norm == "std":
            x_train, x_test = standardize(x_train, x_test)
            
        x_train = build_poly(x_train, poly_degree)
        x_test = build_poly(x_test, poly_degree)
        
        if method == "ridge":
            w = ridge_regression(tx=x_train, y=y_train, lambda_=lambda_)
        #elif: ADD NEW METHODS
        
        train_pred = predict_labels(w, x_train, method)
        train_accuracy.append(np.sum(y_train == train_pred)*1.0/len(y_train))
        
        predictions[Y_inds[i]] = predict_labels(w, x_test, method)
    
    print(train_accuracy)
    print(np.mean(train_accuracy))
    return predictions