import pandas as pd


def split(data, m):
    return data, data


def compare_with_sklearn():
    df = pd.read_csv('dataset/dataset_1_full.txt')
    train, test = split(df, 0.7)
    print train.shape, test.shape


if __name__ == '__main__':
    compare_with_sklearn()
