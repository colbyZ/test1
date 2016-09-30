from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

Dataset_1_Data = namedtuple('Dataset_1_Data', ['x', 'y'])


def load_dataset_1():
    # Load data
    data = np.loadtxt('datasets/dataset_1.txt', delimiter=',', skiprows=1)

    # Split predictors and response
    x = data[:, :-1]
    y = data[:, -1]

    return Dataset_1_Data(x, y)


def heatmap_prob_1a(dataset_1_data):
    x = dataset_1_data.x

    # Compute matrix of correlation coefficients
    corr_matrix = np.corrcoef(x.T)

    # Display heat map
    _, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.pcolor(corr_matrix)
    ax.set_title('Heatmap of correlation matrix')

    plt.show()


def exhaustive_search_prob_1b(dataset_1_data):
    x = dataset_1_data.x

    # Best Subset Selection
    min_bic = float('inf')  # set some initial large value for min BIC score
    best_subset = []  # best subset of predictors

    # Create all possible subsets of the set of 10 predictors
    predictor_set = range(x.shape[1])  # predictor set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    print predictor_set


def main():
    dataset_1_data = load_dataset_1()

    # heatmap_prob_1a(dataset_1_data)
    exhaustive_search_prob_1b(dataset_1_data)


if __name__ == '__main__':
    main()
