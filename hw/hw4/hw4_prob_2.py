from collections import namedtuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression as Lin_Reg
from sklearn.linear_model import Ridge as Ridge_Reg

# prob 2a

Dataset_2_Data = namedtuple('Dataset_2_Data', ['train', 'test'])

XY_Data = namedtuple('XY_Data', ['x', 'y'])


def split(df, split_index):
    train = df[:split_index]
    test = df[split_index:]
    return train, test


def is_categorical(column):
    return column.dtype == object or len(column.unique()) < 8


def encode_categorical_variables_prob_2a():
    df = pd.read_csv('datasets/dataset_2.txt')

    y = df['price']
    x_df = df.drop('price', axis=1)

    expanded_x_df = x_df.copy()
    for column_name in df.columns:
        column = df[column_name]
        if is_categorical(column):
            dummies_df = pd.get_dummies(column, prefix=column_name)

            expanded_x_df = expanded_x_df.drop(column_name, axis=1)
            expanded_x_df = pd.concat([expanded_x_df, dummies_df], axis=1)

    split_index = len(expanded_x_df) // 4

    x_train, x_test = split(expanded_x_df, split_index)
    y_train, y_test = split(y, split_index)

    train = XY_Data(x_train, y_train)
    test = XY_Data(x_test, y_test)

    return Dataset_2_Data(train, test)


def print_expanded_df_prob_2a(dataset_2_data):
    x = dataset_2_data.train.x

    print '%d predictors:' % len(x.columns)
    print x.columns


# prob 2b

def score(regression, train, test):
    train_score = regression.score(*train)
    test_score = regression.score(*test)
    return train_score, test_score


def linear_regression_prob_2b(dataset_2_data):
    train = dataset_2_data.train
    test = dataset_2_data.test

    print 'x train shape: %s' % str(train.x.shape)

    linear_regression = Lin_Reg()
    linear_regression.fit(*train)

    train_score, test_score = score(linear_regression, train, test)

    print 'train R^2: %.3f, test R^2: %.3f' % (train_score, test_score)


# prob 2c

def ridge_regression_prob_2c(dataset_2_data):
    train = dataset_2_data.train
    test = dataset_2_data.test

    alpha_list = []
    train_score_list = []
    test_score_list = []

    for exponent in range(-7, 8):
        alpha = 10 ** exponent

        ridge_regression = Ridge_Reg(alpha=alpha, normalize=True)
        ridge_regression.fit(*train)
        train_score, test_score = score(ridge_regression, train, test)

        print 'alpha: %.0e, train R^2: %.3f, test R^2: % .3f' % (alpha, train_score, test_score)

        train_score_list.append(train_score)
        test_score_list.append(test_score)
        alpha_list.append(alpha)

    _, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(alpha_list, train_score_list, label='train')
    ax.plot(alpha_list, test_score_list, label='test')

    ax.set_xscale('log')
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$R^2$')
    ax.set_title('Ridge regression, train and test scores as functions of $\lambda$')

    ax.legend(loc='lower right')

    plt.show()


# prob 2d

def cross_validation_prob_2d(dataset_2_data):
    train = dataset_2_data.train

    x_fold = train.x
    y_fold = train.y
    num_folds = 5
    kf = KFold(len(x_fold), n_folds=num_folds, shuffle=True, random_state=1090)

    alpha_list = []
    cv_score_list = []
    for exponent in range(-7, 8):
        alpha = 10 ** exponent
        ridge_regression = Ridge_Reg(alpha=alpha)

        test_score_sum = 0.0
        for train_index, test_index in kf:
            x_fold_train = x_fold.iloc[train_index]
            x_fold_test = x_fold.iloc[test_index]

            y_fold_train = y_fold[train_index]
            y_fold_test = y_fold[test_index]

            ridge_regression.fit(x_fold_train, y_fold_train)
            test_score_sum += ridge_regression.score(x_fold_test, y_fold_test)

        cv_score = test_score_sum / num_folds

        alpha_list.append(alpha)
        cv_score_list.append(cv_score)
        print 'alpha: %.0e, cv score: % .3f' % (alpha, cv_score)

    _, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(alpha_list, cv_score_list)

    ax.set_xscale('log')
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('CV $R^2$')
    ax.set_title('Ridge regression, CV $R^2$ score as a function of $\lambda$')

    plt.show()


def main():
    dataset_2_data = encode_categorical_variables_prob_2a()

    # print_expanded_df_prob_2a(dataset_2_data)
    # linear_regression_prob_2b(dataset_2_data)
    ridge_regression_prob_2c(dataset_2_data)
    # cross_validation_prob_2d(dataset_2_data)


if __name__ == '__main__':
    main()
