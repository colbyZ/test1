from collections import namedtuple

import numpy as np

DatasetData = namedtuple('DatasetData', ['x', 'y'])


def loadtxt(file_name):
    return np.loadtxt('datasets/%s' % file_name, delimiter=',', skiprows=1)


def read_dataset3_data():
    data = loadtxt('dataset_3.txt')

    y = data[:, -1]
    x = data[:, 0]

    return DatasetData(x, y)


def compute_aic_and_bic():
    pass


if __name__ == '__main__':
    dataset_3_data = read_dataset3_data()
    compute_aic_and_bic()
