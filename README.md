# Project 1 : Detecting Higgs Boson

This project was created as part of the Machine Learning course [ **CS-433** ] at EPFL. We created a binary classification
model for detecting the presence of Higgs Bosson particle in events from the Large Hadron Collider at CERN. For this purpose, 
several machine learning techniques were implemented and compared. The final model uses regularized logistic regression. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
The project was created and tested with the following dependencies:

```
- Anaconda Python 3.6.5 
- NumPy 1.14.3
```

## Installing

This project does not require any installation. To use, simply clone the repository to your local machine using the following
command:

```
git clone https://github.com/mmilenkoski/ml_project1.git
```

## Project Structure
The project is organized as follows:

    .
    ├── data                     # Train and test datasets
    ├── predictions              # Prediction files for submission on Kaggle
    ├── utils                    # Utilization files for preprocessing, training and evaluation.
    │   ├── helpers.py           # Helper functions for loading data, making predictions, and creating submission files.
    │   ├── hyperparameters.py   # Functions for accessing the best hyperparameters obtained from holdout validation.
    │   ├── implementations.py   # Implementations of required machine learning models.
    │   └── preprocessing.py     # Data cleaning and feature engineering functions.
    ├── README.md                # README file
    └── run.py                   # Script for training the optimal model, and creating a file with final predictions.
    
## Running

Before training the model, please unzip the files `data/train.zip` and `data/test.zip` in the folder `data`. You can also unzip the file `data/sample-submission.zip` in order to see the format of the submissions for Kaggle. To reproduce our results run the following command:

``` 
python run.py
```

After running the script `run.py` you can find the generated predictions in the file `predictions/predictions.csv`. Our final predictions are in the file `predictions/predictions_final.csv` for comparison.

## Tune hyperparameters

We obtained the final hyperparameter using holdout validation and grid search. To train the model with your own hyperparameters, change their values in the file `utils/hyperparameters.py`, and run the script `run.py`

## Authors

* Martin Milenkoski     martin.milenkoski@epfl.ch
* Blagoj Mitrevski      blagoj.mitrevski@epfl.ch
* Samuel Bosch          samuel.bosch@epfl.ch
