import glob
import math
import os
import random
import string

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

WEIGHTS_DIR = 'weights/'


def latest_modified_weight():
    """

    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def generate_model_name(size=5):
    """
    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def load_model(path):
    """

    :param path: weight path
    :return: load model based on the path
    """
    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def train_model(features, labels):
    model = LogisticRegression()
    model.fit(features, labels)
    mse = mean_squared_error(features, labels)
    print(f'MSE {math.sqrt(mse)}')
    print(f'R^2 value: {model.score(features, labels)}')
    print(f'b_0: {model.coef_[0][0]} \nb_1: {model.intercept_[0]}')

    ans = input('Do you want to save the model weight? ')
    if ans in ('yes', '1'):
        model_name = WEIGHTS_DIR + 'LogReg-' + generate_model_name(5) + '.pkl'
        with open(model_name, 'wb') as f:
            joblib.dump(value=model, filename=f, compress=3)
            print(f'Model saved at {model_name}')
    return model


def logistic(model, x):
    """

    :param model: Linear Regression model , b_0 + b_1*x
    :param x:
    :return: 1 / ( 1 + e^( - (b_0 + b_1*x) ) ) Logistic Regression model
    """
    return 1 / (1 + np.exp(-(model.intercept_ + model.coef_ * x)))
