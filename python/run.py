############################################################################
# Group 36 - Chiara Orvati, Sandro Barnabishvili, Michael Heiniger
# This file outputs the best predictions we had on Kaggle.com
# It uses Ridge regression with a polynomial basis (i.e. every feature is
# repeated "degree" times with its value to the power 1,2,...,degree)
# as well as the sinc and cos applied to each feature separately.
############################################################################

import numpy as np
from proj1_helpers import *
from helpers import *
from data_modification import *
from implementations import *
from feature_transformation import *

##################################
# PATHS TO ADAPT:
##################################

# Path of the TRAINING data CSV file
DATA_TRAIN_PATH = '../data/train.csv'

# Path of the TEST data CSV file
DATA_TEST_PATH = '../data/test.csv'

# Path of the predictions CSV file
OUTPUT_PATH = '../data/36-best-predictions.csv'


""" Load and preprocess TRAINING data """
y, raw_tx, ids = load_csv_data(DATA_TRAIN_PATH)

y = y[:,np.newaxis] # Make it a vector shape=(x,) -> shape=(x,1)

# Replace -999 by the mean of its respective column
processed_tx = replace_by_mean(raw_tx)

# Standardize (subtract mean and divide by standard deviation)
processed_tx,mean_pr_tx,std_pr_tx = standardize(processed_tx)

""" Load and preprocess TEST data """
_, raw_tx_test, ids_test = load_csv_data(DATA_TEST_PATH)

processed_tx_test = replace_by_mean(raw_tx_test)
processed_tx_test = standardize_test(processed_tx_test, mean_pr_tx, std_pr_tx)


""" Add functions of features to TRAINING data """

tx_augmented = np.copy(processed_tx)


# Degree of the polynomial for each feature
degree = 11
tx_augmented = build_poly_matrix(tx_augmented, degree)

# Add cos(x1), cos(x2),..., cos(x30) to the feature matrix
tx_augmented = add_cos_function(tx_augmented)

# Add sinc(x1), sinc(x2),..., sinc(x30) to the feature matrix
tx_augmented = add_sinc_function(tx_augmented)


""" Perform Ridge regression """
# Value of the hyper-parameter lambda for the Ridge regression
lambda_ = 1e-6
weights, loss = ridge_regression(y, tx_augmented, lambda_)


""" Add functions of features to TEST data """

tx_test_augmented = np.copy(processed_tx_test)

tx_test_augmented = build_poly_matrix(tx_test_augmented, degree)

# Add cos(x1), cos(x2),..., cos(x30) to the feature matrix
tx_test_augmented = add_cos_function(tx_test_augmented)

# Add sinc(x1), sinc(x2),..., sinc(x30) to the feature matrix
tx_test_augmented = add_sinc_function(tx_test_augmented)


""" Generate predictions for TEST data """

# Generate predictions
y_pred = predict_labels(weights, tx_test_augmented)

# Output predictions in csv file
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)