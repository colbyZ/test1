from itertools import izip

import numpy as np
from sklearn.linear_model import LinearRegression as Lin_Reg


def polynomial_regression_score(y_predicted, y_test):
    rss = 0.0
    tss = 0.0
    y_mean = np.mean(y_test)
    for predicted_value, actual_value in izip(y_predicted, y_test):
        rss += (actual_value - predicted_value) ** 2
        tss += (actual_value - y_mean) ** 2

    r_squared = 1.0 - rss / tss
    return r_squared, rss


def loadtxt(file_name):
    return np.loadtxt('datasets/%s' % file_name, delimiter=',', skiprows=1)


def split_y_x(data):
    y = data[:, -1]
    x = data[:, :-1]
    return y, x


# prob 4a


def get_new_column(x_train, degree_pair):
    return np.prod(x_train ** degree_pair, axis=1)


def generate_variable_degrees(degree_of_the_polynomial):
    pair_list = []
    for max_degree in xrange(1, degree_of_the_polynomial + 1):
        for degree1 in xrange(max_degree + 1):
            degree2 = max_degree - degree1
            pair_list.append((degree1, degree2))
    return pair_list


def polynomial_regression_fit_prob_4a(x_train, y_train, degree_of_the_polynomial):
    degree_pair_list = generate_variable_degrees(degree_of_the_polynomial)
    poly_columns = [get_new_column(x_train, degree_pair)
                    for degree_pair in degree_pair_list]

    poly_x = np.array(poly_columns).T

    linear_regression = Lin_Reg()
    linear_regression.fit(poly_x, y_train)

    return linear_regression.coef_, linear_regression.intercept_, degree_pair_list


def calculate_polynomial_value_prob_4a(coefs, intercept, xs, degree_pair_list):
    value = intercept

    for degree_pair, coef in izip(degree_pair_list, coefs):
        m = coef
        for degree, x in izip(degree_pair, xs):
            m *= pow(x, degree)
        value += m

    return value


def polynomial_regression_predict_prob_4a(coefs, intercept, x_test, degree_pair_list):
    return [calculate_polynomial_value_prob_4a(coefs, intercept, x, degree_pair_list) for x in x_test]


def polynomial_regression_prob_4a():
    data_train = loadtxt('dataset_1_train.txt')
    data_test = loadtxt('dataset_1_test.txt')

    y_train, x_train = split_y_x(data_train)
    y_test, x_test = split_y_x(data_test)

    for degree in range(1, 4):
        coefs, intercept, degree_pair_list = polynomial_regression_fit_prob_4a(x_train, y_train, degree)
        y_predicted = polynomial_regression_predict_prob_4a(coefs, intercept, x_test, degree_pair_list)
        r_squared, _ = polynomial_regression_score(y_predicted, y_test)

        print 'degree: %d, R^2: %.3f' % (degree, r_squared)


def main():
    polynomial_regression_prob_4a()


if __name__ == '__main__':
    main()
