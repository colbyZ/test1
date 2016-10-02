import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def split(data, split_index):
    train = data[:split_index]
    test = data[split_index:]
    return train, test


# Fit

def get_aug_x_y(reg_param, x_train, y_train):
    n = len(x_train)
    num_predictors = x_train.shape[1]
    aug_x_train = np.concatenate((x_train, np.sqrt(reg_param) * np.identity(n)))
    aug_y_train = np.concatenate((y_train, np.zeros(num_predictors))).reshape(-1, 1)
    return aug_x_train, aug_y_train


def ridge(x_train, y_train, reg_param):
    aug_x_train, aug_y_train = get_aug_x_y(reg_param, x_train, y_train)

    model = LinearRegression()
    model.fit(aug_x_train, aug_y_train)

    return model


# Score

def score(model, x, y, reg_param):
    aug_x, aug_y = get_aug_x_y(reg_param, x, y)
    return model.score(aug_x, aug_y)


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
        model = ridge(x_train, y_train, alpha)
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

    plt.legend()

    plt.show()


def main():
    ridge_regression_prob_3b()


if __name__ == '__main__':
    main()
