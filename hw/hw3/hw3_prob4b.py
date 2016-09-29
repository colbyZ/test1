import numpy as np


# Functions for fitting and evaluating multiple linear regression

# --------  multiple_linear_regression_fit
# A function for fitting a multiple linear regression
# Fitted model: f(x) = x.w + c
# Input:
#      x_train (n x d array of predictors in training data)
#      y_train (n x 1 array of response variable vals in training data)
# Return:
#      w (d x 1 array of coefficients)
#      c (float representing intercept)

def multiple_linear_regression_fit(x_train, y_train):
    # Append a column of one's to x
    ones_col = np.ones_like(y_train).reshape(-1, 1)
    x_train = np.concatenate((x_train, ones_col), axis=1)

    # Compute transpose of x
    x_transpose = x_train.T

    # Compute coefficients: w = inv(x^T * x) x^T * y
    # Compute intermediate term: inv(x^T * x)
    # Note: We have to take pseudo-inverse (pinv), just in case x^T * x is not invertible
    x_t_x_inv = np.linalg.pinv(x_transpose.dot(x_train))

    # Compute w: inter_term * x^T * y
    w = x_t_x_inv.dot(x_transpose).dot(y_train)

    # Obtain intercept: 'c' (last index)
    c = w[-1]

    return w[:-1], c


# --------  multiple_linear_regression_score
# A function for evaluating R^2 score and MSE
# of the linear regression model on a data set
# Input:
#      w (d x 1 array of coefficients)
#      c (float representing intercept)
#      x_test (n x d array of predictors in testing data)
#      y_test (n x 1 array of response variable vals in testing data)
# Return:
#      r_squared (float)
#      y_pred (n x 1 array of predicted y-vals)

def multiple_linear_regression_score(w, c, x_test, y_test):
    # Compute predicted labels
    y_pred = x_test.dot(w) + c

    # Evaluate squared error, against target labels
    # sq_error = \sum_i (y[i] - y_pred[i])^2
    sq_error = np.square(y_test - y_pred).sum()

    # Evaluate squared error for a predicting the mean value, against target labels
    # variance = \sum_i (y[i] - y_mean)^2
    y_mean = y_test.mean()
    y_variance = np.square(y_test - y_mean).sum()

    # Evaluate R^2 score value
    r_squared = 1 - sq_error / y_variance

    return r_squared, y_pred


def loadtxt(file_name, dtype=float):
    return np.loadtxt('datasets/%s' % file_name, delimiter=',', skiprows=1, dtype=dtype)


def split_y_x(data):
    y = data[:, -1]
    x = data[:, :-1]
    return y, x


# prob 4b

def multiple_weighted_linear_regression_fit(x_train, y_train, weight_matrix):
    ones_col = np.ones_like(y_train).reshape(-1, 1)
    x_train = np.concatenate((x_train, ones_col), axis=1)

    x_transpose = x_train.T

    x_t_w_x_inv = np.linalg.pinv(x_transpose.dot(weight_matrix).dot(x_train))

    w = x_t_w_x_inv.dot(x_transpose).dot(weight_matrix).dot(y_train)

    c = w[-1]

    return w[:-1], c


def weighted_linear_regression_prob_4b():
    data_train = loadtxt('dataset_1_train.txt')
    data_test = loadtxt('dataset_1_test.txt')
    data_train_noise_levels = loadtxt('dataset_1_train_noise_levels.txt', dtype=str)

    y_train, x_train = split_y_x(data_train)
    y_test, x_test = split_y_x(data_test)

    w, c = multiple_linear_regression_fit(x_train, y_train)
    r_squared, _ = multiple_linear_regression_score(w, c, x_test, y_test)

    print 'plain linear regression, test R^2: %.3f' % r_squared

    for noise_weight in np.arange(0.0, 1.0, step=0.05):
        weights = [1.0 if noise_level_str == 'none' else noise_weight for noise_level_str in data_train_noise_levels]

        weight_matrix = np.diag(weights)

        coefs, intercept = multiple_weighted_linear_regression_fit(x_train, y_train, weight_matrix)
        r_squared, _ = multiple_linear_regression_score(coefs, intercept, x_test, y_test)

        print 'weighted linear regression, noise weight: %.2f, test R^2: %.3f' % (noise_weight, r_squared)


def main():
    weighted_linear_regression_prob_4b()


if __name__ == '__main__':
    main()
