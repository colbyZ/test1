from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression as Lin_Reg

DatasetData = namedtuple('DatasetData', ['x', 'y'])


def loadtxt(file_name):
    return np.loadtxt('datasets/%s' % file_name, delimiter=',', skiprows=1)


def split_y_x(data):
    y = data[:, -1]
    x = data[:, :-1]
    return y, x


def read_dataset_data(filename):
    # Load train set
    data = loadtxt(filename)

    y, x = split_y_x(data)

    return DatasetData(x, y)


def polynomial_regression_fit(x_train, y_train, degree_of_the_polynomial):
    poly_x = x_train
    for exponent in range(2, degree_of_the_polynomial + 1):
        new_column = np.power(x_train, exponent)
        poly_x = np.hstack((poly_x, new_column))

    linear_regression = Lin_Reg()
    linear_regression.fit(poly_x, y_train)

    return linear_regression.coef_, linear_regression.intercept_


def calculate_polynomial_value(x, coefs, intercept):
    poly_sum = intercept
    for i, coef in enumerate(coefs):
        poly_sum += coef * pow(x, i + 1)
    return poly_sum


def calculate_polynomial_values(xs, coefs, intercept):
    return [calculate_polynomial_value(x, coefs, intercept) for x in xs]


def fit_and_visualize_prob_2a():
    x, y = dataset_3_data

    degrees = [3, 5, 10, 25]

    degrees_len = len(degrees)
    fig, axes = plt.subplots(degrees_len, 1, figsize=(12, 6 * degrees_len))

    xs = np.linspace(0.01, 0.99)
    for i, degree in enumerate(degrees):
        ax = axes[i]
        coefs, intercept = polynomial_regression_fit(x, y, degree)
        ax.scatter(x, y, color='blue')

        ax.plot(xs, calculate_polynomial_values(xs, coefs, intercept), color='red')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('degree of the polynomial: %d' % degree)

    plt.show()


def train_test_split_by_index(data, index):
    train = data[:index]
    test = data[index:]
    return train, test


def compare_errors_prob_2b():
    x, y = dataset_3_data
    mid_index = len(x) / 2

    x_train, x_test = train_test_split_by_index(x, mid_index)
    y_train, y_test = train_test_split_by_index(y, mid_index)

    print y_train


if __name__ == '__main__':
    dataset_3_data = read_dataset_data("dataset_3.txt")
    # fit_and_visualize_prob_2a()
    compare_errors_prob_2b()
