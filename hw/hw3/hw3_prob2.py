import numpy as np
from sklearn.linear_model import LinearRegression as Lin_Reg


def loadtxt(file_name):
    return np.loadtxt('datasets/%s' % file_name, delimiter=',', skiprows=1)


def split(data):
    y = data[:, -1]
    x = data[:, :-1]
    return y, x


def polynomial_regression_fit(x_train, y_train, degree_of_the_polynomial):
    poly_x = x_train
    for exponent in range(2, degree_of_the_polynomial + 1):
        new_column = np.power(x_train, exponent)
        poly_x = np.hstack((poly_x, new_column))

    linear_regression = Lin_Reg()
    linear_regression.fit(poly_x, y_train)

    return linear_regression.coef_, linear_regression.intercept_


def fit_and_visualize_prob_2b():
    data = loadtxt("dataset_3.txt")
    y, x = split(data)

    coefs, intercept = polynomial_regression_fit(x, y, 3)


if __name__ == '__main__':
    fit_and_visualize_prob_2b()
