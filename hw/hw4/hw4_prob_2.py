from collections import namedtuple

import pandas as pd
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

    print '%d columns:' % len(x.columns)
    print x.columns


# prob 2b

def score(regression, train, test):
    train_score = regression.score(*train)
    test_score = regression.score(*test)
    return train_score, test_score


def linear_regression_prob_2b(dataset_2_data):
    train = dataset_2_data.train
    test = dataset_2_data.test

    linear_regression = Lin_Reg()
    linear_regression.fit(*train)

    train_score, test_score = score(linear_regression, train, test)

    print 'train R^2: %.3f, test R^2: %.3f' % (train_score, test_score)


# prob 2c

def ridge_regression_prob_2c(dataset_2_data):
    train = dataset_2_data.train
    test = dataset_2_data.test

    for exponent in range(-7, 8):
        alpha = 10 ** exponent

        ridge_regression = Ridge_Reg(alpha=alpha)
        ridge_regression.fit(*train)
        train_score, test_score = score(ridge_regression, train, test)

        print 'alpha: %.0e, train R^2: %.3f, test R^2: %.3f' % (alpha, train_score, test_score)


def main():
    dataset_2_data = encode_categorical_variables_prob_2a()

    # print_expanded_df_prob_2a(dataset_2_data)
    # linear_regression_prob_2b(dataset_2_data)
    ridge_regression_prob_2c(dataset_2_data)


if __name__ == '__main__':
    main()
