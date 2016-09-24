from collections import namedtuple
from itertools import izip

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
    x_column = np.vstack(x_train)
    poly_x = x_column
    for exponent in range(2, degree_of_the_polynomial + 1):
        new_column = np.power(x_column, exponent)
        poly_x = np.hstack((poly_x, new_column))

    linear_regression = Lin_Reg()
    linear_regression.fit(poly_x, y_train)

    return linear_regression.coef_, linear_regression.intercept_


def calculate_polynomial_value(coefs, intercept, x):
    poly_sum = intercept
    for i, coef in enumerate(coefs):
        poly_sum += coef * pow(x, i + 1)
    return poly_sum


def polynomial_regression_predict(coefs, intercept, degree_of_the_polynomial, x_test):
    return [calculate_polynomial_value(coefs, intercept, x) for x in x_test]


def polynomial_regression_score(y_predicted, y_test):
    rss = 0.0
    tss = 0.0
    y_mean = np.mean(y_test)
    for predicted_value, actual_value in izip(y_predicted, y_test):
        rss += (actual_value - predicted_value) ** 2
        tss += (actual_value - y_mean) ** 2

    r_squared = 1.0 - rss / tss
    return r_squared, rss


def read_dataset3_data():
    data = loadtxt('dataset_3.txt')

    y = data[:, -1]
    x = data[:, 0]

    return DatasetData(x, y)


def fit_and_visualize_prob_2a():
    x, y = dataset_3_data

    degrees = [3, 5, 10, 25]

    degrees_len = len(degrees)
    _, axes = plt.subplots(degrees_len, 1, figsize=(12, 6 * degrees_len))

    xs = np.linspace(0.01, 0.99)
    for i, degree in enumerate(degrees):
        ax = axes[i]
        coefs, intercept = polynomial_regression_fit(x, y, degree)
        ax.scatter(x, y, color='blue')

        ax.plot(xs, polynomial_regression_predict(coefs, intercept, degree, xs), color='red')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('degree of the polynomial: %d' % degree)

    plt.show()


def evaluate_polynomial_regression_fit(coefs, intercept, x_test, y_test, degree):
    y_predicted = polynomial_regression_predict(coefs, intercept, degree, x_test)
    return polynomial_regression_score(y_predicted, y_test)


def train_test_split_by_index(data, index):
    train = data[:index]
    test = data[index:]
    return train, test


def compare_errors_prob_2b():
    x, y = dataset_3_data
    mid_index = len(x) / 2

    x_train, x_test = train_test_split_by_index(x, mid_index)
    y_train, y_test = train_test_split_by_index(y, mid_index)

    _, ax = plt.subplots(1, 1, figsize=(12, 6))

    r_sq_train_list = []
    r_sq_test_list = []
    degrees = range(1, 16)
    for degree in degrees:
        coefs, intercept = polynomial_regression_fit(x_train, y_train, degree)
        r_sq_train, _ = evaluate_polynomial_regression_fit(coefs, intercept, x_train, y_train, degree)
        r_sq_train_list.append(r_sq_train)
        r_sq_test, _ = evaluate_polynomial_regression_fit(coefs, intercept, x_test, y_test, degree)
        r_sq_test_list.append(r_sq_test)
        print 'degree: %2d, train R^2: %.4f, test R^2: %.4f' % (degree, r_sq_train, r_sq_test)

    ax.plot(degrees, r_sq_train_list, label='train')
    ax.plot(degrees, r_sq_test_list, label='test')
    ax.legend(loc='lower right')
    ax.set_xlabel('degree of the polynomial')
    ax.set_ylabel('$R^2$')

    plt.show()


if __name__ == '__main__':
    dataset_3_data = read_dataset3_data()
    # fit_and_visualize_prob_2a()
    compare_errors_prob_2b()
