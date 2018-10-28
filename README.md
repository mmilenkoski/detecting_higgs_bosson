# Project 1 : Predicting Higgs Bosson

This project was created as part of the Machine Learning course [ **CS-433** ] at EPFL. We created a binary classification
model for detecting the presence of Higgs Bosson particle in events from the Large Hadron Collider at CERN. For this purpose, 
several machine learning techniques were implemented and compared. The final model uses regularized logistic regression. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
For running the project you will need the following dependencies:

```
- Python 3.6+ 
- NumPy 1.14+
```

## Installing

This project does not require any installation. To use, simply clone the repository to your local machine using the following
command:

```
git clone https://github.com/mmilenkoski/ml_project1.git
```

## Project Structure

    .
    ├── utils                    # Utilization files for preprocessing, training and evaluation
    │   ├── helpers.py           # Helper functions for loading data, making predictions, and creating submission files
    │   ├── hyperparameters.py   # Functions for accessing the best hyperparameters obtained from holdout validation
    │   ├── implementations.py   # Implementations of required machine learning models and helper functions for them
    │   └── preprocessing.py     # Data cleaning and feature engineering functions
    └── run.py                   # File for training the optimal model and creating a file with final predictions.
