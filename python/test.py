import numpy as np
from cross_validation import *
from proj1_helpers import *


DATA_TRAIN_PATH = '/Users/chiara/Documents/EPFL/Master/Ma1/MachineLearning/project1/data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

lambdas = np.logspace(-4, 2, 30)
degrees = np.arange(1,10)
k_fold = 4

def main():
    l, err = get_best_lambda(y, tX[:,0], 4, k_fold, lambdas)
    print(l)
    print(err)

main()