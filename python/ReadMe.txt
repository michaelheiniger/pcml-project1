###########################################################################################
# PCML course - Project 1: Chiara Orvati, Sandro Barnabishvili, Michael Heiniger
# The purpose of this file is to present the main files of the Python code written 
# and used for the project.
###########################################################################################


1. Executable files:
- run.py does the regression on the training set using the best model we have found and compute the predictions on the test set. 
- Ridge_regression_notebook.pynb is a Python notebook built as a framework to use and
 assess Ridge regression. To assess the different models, it uses k-fold cross-validation and outputs plots of the error rate versus a wide range of values of lambda.


2. Functions files:
- implementations.py contains the 6 functions implementing the different regression methods. It contains all its dependencies to make it self-contained (except for numpy) to make it painless for the teaching team to run.

- data_modification.py contains a function used to replace the values "-999" in the dataset by the mean of the feature (mean computed ignoring the "-999" values). This is one way to deal with these meaningless values.

- feature_transformation.py contains all functions used to augment the feature matrix in order to find a more suitable model than the basic features.

- proj1_helpers.py contains useful functions to load the data from CSV files, output predictions of unlabeled data based on computed weights, and create predictions CSV submission files.

- helpers.py contains useful functions to implement cross-validation methods (split in two based on a ratio, k-fold), standardize data and iterate over datasets.

- plots.py contains functions that outputs plots used to assess our models.

- costs.py groups loss functions for MSE

- cross_validation_RLR.py contains functions to do k-fold cross-validation for regularized logistic regression

- run_ridge_regression.py contains functions to do k-fold cross-validation for ridge regression




