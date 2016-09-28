import numpy as np


def loadtxt(file_name):
    return np.loadtxt('datasets/%s' % file_name, delimiter=',', skiprows=1)


def prob_4a():
    data_train = loadtxt('dataset_1_train.txt')
    data_test = loadtxt('dataset_1_test.txt')

    print data_train.shape, data_test.shape


def main():
    prob_4a()


if __name__ == '__main__':
    main()
