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
    n = x_train.shape[0]
    ones_col = np.ones((n, 1))
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


def loadtxt(file_name):
    return np.loadtxt('datasets/%s' % file_name, delimiter=',', skiprows=1)


def split(data):
    y = data[:, -1]
    x = data[:, :-1]
    return y, x


def evaluate_model_prob_1a():
    # Load train and test data sets
    data_train = loadtxt('dataset_1_train.txt')
    data_test = loadtxt('dataset_1_test.txt')

    # Split predictors from response
    # Training
    y_train, x_train = split(data_train)

    # Testing
    y_test, x_test = split(data_test)

    # Fit multiple linear regression model
    w, c = multiple_linear_regression_fit(x_train, y_train)

    # Evaluate model
    r_squared, _ = multiple_linear_regression_score(w, c, x_test, y_test)

    print 'R^2 score on test set: %.3f' % r_squared


if __name__ == '__main__':
    evaluate_model_prob_1a()
