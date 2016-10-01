from collections import namedtuple

import pandas as pd
from sklearn.linear_model import LinearRegression as Lin_Reg

# prob 2a

Dataset_2_Data = namedtuple('Dataset_2_Data', ['x', 'y'])


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

    return Dataset_2_Data(expanded_x_df, y)


def print_expanded_df_prob_2a(dataset_2_data):
    x = dataset_2_data.x

    print '%d columns:' % len(x.columns)
    print x.columns


# prob 2b

def split(df, split_index):
    train = df[:split_index]
    test = df[split_index:]
    return train, test


def linear_regression_prob_2b(dataset_2_data):
    x = dataset_2_data.x
    y = dataset_2_data.y

    split_index = len(x) // 4

    x_train, x_test = split(x, split_index)
    y_train, y_test = split(y, split_index)

    linear_regression = Lin_Reg()
    linear_regression.fit(x_train, y_train)

    train_score = linear_regression.score(x_train, y_train)
    test_score = linear_regression.score(x_test, y_test)

    print 'train R^2: %.3f, test R^2: %.3f' % (train_score, test_score)


def main():
    dataset_2_data = encode_categorical_variables_prob_2a()

    # print_expanded_df_prob_2a(dataset_2_data)
    linear_regression_prob_2b(dataset_2_data)


if __name__ == '__main__':
    main()
