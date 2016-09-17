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


def compare_with_sklearn():
    np.random.seed(1090)
    df = pd.read_csv('dataset/dataset_1_full.txt')
    train, test = split(df, 0.7)
    print train.shape, test.shape
    print train.head()


if __name__ == '__main__':
    compare_with_sklearn()
