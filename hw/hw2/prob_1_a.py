import numpy as np
import pandas as pd


def split(data, m):
    test_size = 1 - m
    length = len(data)

    indices = list(data.index)
    np.random.shuffle(indices)

    test_length = int(round(test_size * length))
    test_indices = indices[:test_length]
    train_indices = indices[test_length:]

    test = data.loc[test_indices]
    train = data.loc[train_indices]

    return train, test


def knn_predict_one_point(k, sorted_train, test_x):
    return 0.0


def knn_predict(k, train, test):
    sorted_train = train.sort_values(by='x')
    predicted_test = test.copy()
    predicted_values = [knn_predict_one_point(k, sorted_train, row['x']) for index, row in test.iterrows()]
    predicted_test['y'] = predicted_values
    return predicted_test


def compare_with_sklearn():
    np.random.seed(1090)

    df = pd.read_csv('dataset/dataset_1_full.txt')

    train, test = split(df, 0.7)

    predicted_test = knn_predict(1, train, test[['x']])
    print predicted_test


if __name__ == '__main__':
    compare_with_sklearn()
