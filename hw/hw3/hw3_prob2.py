import numpy as np


def loadtxt(file_name):
    return np.loadtxt('datasets/%s' % file_name, delimiter=',', skiprows=1)


def polynomial_regression_fit(x_train, y_train, degree_of_the_polynomial):
    pass


def fit_polynomial_regression_models_prob_2b():
    data = loadtxt("dataset_3.txt")

    print data


if __name__ == '__main__':
    fit_polynomial_regression_models_prob_2b()
