from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

DatasetData = namedtuple('DatasetData', ['x', 'y'])


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


def read_dataset_data(filename):
    # Load train set
    data = loadtxt(filename)

    y, x = split(data)

    return DatasetData(x, y)


def plot_histograms_prob_1b():
    np.random.seed(1090)

    x, y = dataset_2_data

    # Record size of the data set
    n = x.shape[0]
    d = x.shape[1]
    subsample_size = 100

    # No. of subsamples
    num_samples = 200

    # Linear regression with all 5 predictors

    # Create a n x d array to store coefficients for 100 subsamples
    coefs_multiple = np.zeros((num_samples, d))

    print 'Linear regression with all predictors'

    # Repeat for 200 subsamples
    for i in range(num_samples):
        # Generate a random subsample of <subsample_size> data points
        random_indices = np.random.choice(n, subsample_size)
        x_subsample = x[random_indices, :]
        y_subsample = y[random_indices]

        # Fit linear regression model on subsample
        w, c = multiple_linear_regression_fit(x_subsample, y_subsample)

        # Store the coefficient for the model we obtain
        coefs_multiple[i, :] = w

    # Plot histogram of coefficients, and report their confidence intervals
    fig, axes = plt.subplots(1, d, figsize=(20, 3))

    # Repeat for each coefficient
    for j in range(d):
        # Compute mean for the j-th coefficient from subsamples
        coef_j_mean = coefs_multiple[:, j].mean()

        # Compute confidence interval at 95% confidence level
        conf_int_left = np.percentile(coefs_multiple[:, j], 2.5)
        conf_int_right = np.percentile(coefs_multiple[:, j], 97.5)

        # Plot histogram of coefficient values
        ax = axes[j]
        ax.hist(coefs_multiple[:, j], 15, alpha=0.5)

        # Plot vertical lines at mean and left, right extremes of confidence interval
        ax.axvline(x=coef_j_mean, linewidth=3)
        ax.axvline(x=conf_int_left, linewidth=1, c='r')
        ax.axvline(x=conf_int_right, linewidth=1, c='r')

        # Set plot labels
        ax.set_title('[%.4f, %.4f]' % (conf_int_left, conf_int_right))
        ax.set_xlabel('Predictor %d' % (j + 1))
        ax.set_ylabel('Frequency')

    plt.show()


def compute_confidence_intervals_prob_1b():
    x, y = dataset_2_data
    d = x.shape[1]

    # Add column of ones to x matrix
    x = sm.add_constant(x)

    # Create model for linear regression
    model = sm.OLS(y, x)
    # Fit model
    fitted_model = model.fit()
    # The confidence intervals for our five coefficients are contained in the last five
    # rows of the fitted_model.conf_int() array
    conf_int = fitted_model.conf_int()[1:, :]

    for j in range(d):
        print 'the confidence interval for coefficient %d: [%.4f, %.4f]' % (j + 1, conf_int[j][0], conf_int[j][1])


if __name__ == '__main__':
    # evaluate_model_prob_1a()

    dataset_2_data = read_dataset_data("dataset_2.txt")
    plot_histograms_prob_1b()
    # compute_confidence_intervals_prob_1b()
