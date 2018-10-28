# -*- coding: utf-8 -*-
import numpy as np
from utils.helpers import *
from utils.preprocessing import *
from utils.implementations import *
from utils.hyperparameters import *

# set seed for reproducibility
np.random.seed(5)

# File paths
train_file = "./data/train.csv"
test_file = "./data/test.csv"
output_file = "./predictions/predictions.csv"

# Load data
print("Loading data...")
Y_train, X_train, Ids_train = load_csv_data(train_file)
Y_test, X_test, Ids_test = load_csv_data(test_file)

# Split data into eight datasets
print("Splitting data...")
x_train_split, y_train_split, ids_train_split, indx_train_split = split_data_in_eight(X_train, Y_train, Ids_train)
x_test_split, y_test_split, ids_test_split, indx_test_split = split_data_in_eight(X_test, Y_test, Ids_test)

# Adjust angular features
print("Adjusting data...")
x_train_split = adjust_cartesian_features(x_train_split)
x_test_split = adjust_cartesian_features(x_test_split)

# Load hyperparameters
number_of_models = get_number_of_models()
max_iters = get_max_iters()

predictions = Y_test.copy()
train_accuracy = []

print("Training %s models..." % (number_of_models))
for model in range(number_of_models):
    print("Training model: %s/%s" % (model+1, number_of_models))
    # Data for current model
    x_train = x_train_split[model].copy()
    y_train = y_train_split[model]
    x_test = x_test_split[model].copy()
    y_test = y_test_split[model]
    
    # transform labels (-1 -> 0, 1 -> 1) to train the logistic regressions
    y_train = transform_labels_to_zero_one(y_train)
    
    # standardize the train and test data
    x_train, x_test = standardize(x_train, x_test)
    
    #expand the data with polynomial features
    poly_degree = get_poly_degree(model)
    x_train = build_poly(x_train, poly_degree)
    x_test = build_poly(x_test, poly_degree)
    
    # initialize weights for the logistic regression
    initial_w = np.random.randn(x_train.shape[1])
    print (initial_w)
    # train with logistic regression
    w, _ = reg_logistic_regression(y=y_train, tx=x_train, initial_w=initial_w, max_iters=max_iters, gamma=get_gamma(model), lambda_=get_lambda(model))
    print (w)
    print()
    # predict the labes for the test data
    train_pred = predict_labels(w, x_train, "logistic")
    
    # transform labels (-1 -> 0, 1 -> 1) for evaluation
    train_pred = transform_labels_to_zero_one(train_pred)
    
    # compute the classification accuracy for the current model
    train_accuracy.append(accuracy_score(train_pred, y_train))
    
    # predict the labels for the training data
    test_pred = predict_labels(w, x_test, "logistic")
    test_pred.shape= (-1, )
    predictions[indx_test_split[model]] = test_pred

print("Train accuracy for every model: %s" % (["%.2f" % x for x in train_accuracy]))
print("Mean train accuracy: %.2f" % (np.mean(train_accuracy)))

# save the predictions to the output_file
print("Saving csv for submission...")
create_csv_submission(Ids_test, predictions, output_file)
print("Everything done.")
