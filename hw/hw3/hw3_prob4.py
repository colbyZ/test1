import numpy as np
from sklearn.linear_model import LinearRegression as Lin_Reg


def loadtxt(file_name):
    return np.loadtxt('datasets/%s' % file_name, delimiter=',', skiprows=1)


def split_y_x(data):
    y = data[:, -1]
    x = data[:, :-1]
    return y, x


def add_column(poly_x, new_column):
    return np.hstack((poly_x, new_column))


def multiply_columns(x_columns, m1, m2):
    col1 = np.power(x_columns[0], m1)
    col2 = np.power(x_columns[1], m2)
    return np.multiply(col1, col2)


def polynomial_regression_fit_prob_4a(x_train, y_train, degree_of_the_polynomial):
    x_columns = [np.vstack(x_train[:, i]) for i in range(2)]

    poly_x = x_train.copy()

    for exponent in range(2, degree_of_the_polynomial + 1):
        for i in range(2):
            poly_x = add_column(poly_x, np.power(x_columns[i], exponent))
        for m1 in range(1, exponent):
            m2 = exponent - m1
            poly_x = add_column(poly_x, multiply_columns(x_columns, m1, m2))

    linear_regression = Lin_Reg()
    linear_regression.fit(poly_x, y_train)

    return linear_regression.coef_, linear_regression.intercept_


def prob_4a():
    data_train = loadtxt('dataset_1_train.txt')
    data_test = loadtxt('dataset_1_test.txt')

    y_train, x_train = split_y_x(data_train)
    y_test, x_test = split_y_x(data_test)

    coef, intercept = polynomial_regression_fit_prob_4a(x_train, y_train, 3)
    print coef, intercept


def main():
    prob_4a()


if __name__ == '__main__':
    main()
