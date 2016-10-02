from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

RidgeRegressionModel = namedtuple('RidgeRegressionModel', ['linear_regression', 'x_scaler'])


def split(data, split_index):
    train = data[:split_index]
    test = data[split_index:]
    return train, test


# Fit

def get_aug_x_y(reg_param, x, y):
    n = len(x)
    num_predictors = x.shape[1]
    aug_x = np.concatenate((x, np.sqrt(reg_param) * np.identity(n)))
    aug_y = np.concatenate((y, np.zeros(num_predictors))).reshape(-1, 1)
    return aug_x, aug_y


def get_ridge_regression(x_train, y_train, reg_param):
    x_scaler = StandardScaler(with_mean=False)
    x_scaler.fit(x_train)

    scaled_x_train = x_scaler.transform(x_train)

    aug_x_train, aug_y_train = get_aug_x_y(reg_param, scaled_x_train, y_train)

    linear_regression = LinearRegression()
    linear_regression.fit(aug_x_train, aug_y_train)

    return RidgeRegressionModel(linear_regression, x_scaler)


# Score

def score(model, x, y, reg_param):
    scaled_x = model.x_scaler.transform(x)

    aug_x, aug_y = get_aug_x_y(reg_param, scaled_x, y)
    return model.linear_regression.score(aug_x, aug_y)


def ridge_regression_prob_3b():
    # Load
    data = np.loadtxt('datasets/dataset_3.txt', delimiter=',')
    n = len(data)
    split_index = n // 2

    x = data[:, :-1]
    y = data[:, -1]

    x_train, x_test = split(x, split_index)
    y_train, y_test = split(y, split_index)

    # Params
    alphas = [10.0 ** i for i in range(-2, 3)]

    train_score_list = []
    test_score_list = []

    for alpha in alphas:
        model = get_ridge_regression(x_train, y_train, alpha)
        train_score = score(model, x_train, y_train, alpha)
        test_score = score(model, x_test, y_test, alpha)

        train_score_list.append(train_score)
        test_score_list.append(test_score)

        print 'alpha: %.0e, train R^2: %.3f, test R^2: %.3f' % (alpha, train_score, test_score)

    # Plot
    plt.plot(alphas, train_score_list, label='train')
    plt.plot(alphas, test_score_list, label='test')

    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('$R^2$')
    plt.title('train and test R^2 scores as a function of the regularization parameter')
    plt.ylim([-5.0, 1.3])

    plt.legend(loc='lower right')

    plt.show()


def main():
    ridge_regression_prob_3b()


if __name__ == '__main__':
    main()
