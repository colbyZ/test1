import numpy as np

from sklearn.preprocessing import LabelEncoder


def loadtxt(file_name, dtype=float):
    return np.loadtxt('datasets/%s' % file_name, delimiter=',', skiprows=1, dtype=dtype)


# prob 4b

def weighted_linear_regression_prob_4b():
    data_train = loadtxt('dataset_1_train.txt')
    data_test = loadtxt('dataset_1_test.txt')
    data_train_noise_levels = loadtxt('dataset_1_train_noise_levels.txt', dtype=str)

    weights = [1.0 if noise_level_str == 'none' else 0.5 for noise_level_str in data_train_noise_levels]

    print weights

    print data_train.shape, data_test.shape, data_train_noise_levels.shape


def main():
    weighted_linear_regression_prob_4b()


if __name__ == '__main__':
    main()
