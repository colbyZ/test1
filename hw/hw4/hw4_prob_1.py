import itertools as it
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# prob 1a

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


# prob 1b

def get_regression_results(predictor_subset, x, y):
    # Use only a subset of predictors in the training data
    x_subset = x[:, predictor_subset]

    # Fit and evaluate R^2
    model = sm.OLS(y, sm.add_constant(x_subset))

    return model.fit()


def get_best_k_subset(predictor_set, size_k, x, y):
    # Create all possible subsets of size 'size',
    # using the 'combination' function from the 'itertools' library
    subsets_of_size_k = it.combinations(predictor_set, size_k)

    max_r_squared = -float('inf')  # set some initial small value for max R^2 score
    best_k_subset = []  # best subset of predictors of size k
    best_results = None

    # Iterate over all subsets of our predictor set
    for predictor_subset in subsets_of_size_k:
        results = get_regression_results(predictor_subset, x, y)
        r_squared = results.rsquared

        # Update max R^2 and best predictor subset of size k
        # If current predictor subset has a higher R^2 score than that of the best subset
        # we've found so far, remember the current predictor subset as the best!
        if r_squared > max_r_squared:
            max_r_squared = r_squared
            best_k_subset = predictor_subset
            best_results = results

    return best_k_subset, best_results


def exhaustive_search_prob_1b(dataset_1_data):
    x = dataset_1_data.x
    y = dataset_1_data.y

    # Best Subset Selection
    min_bic = float('inf')  # set some initial large value for min BIC score
    best_subset = []  # best subset of predictors
    best_results = None

    num_predictors = x.shape[1]

    # Create all possible subsets of the set of <num_predictors> predictors
    predictor_set = range(num_predictors)  # e.g. predictor set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Repeat for every possible size of subset
    for size_k in range(1, num_predictors + 1):
        k_subset, results = get_best_k_subset(predictor_set, size_k, x, y)

        bic = results.bic

        # Update minimum BIC and best predictor subset
        # If current predictor has a lower BIC score than that of the best subset
        # we've found so far, remember the current predictor as the best!
        if bic < min_bic:
            min_bic = bic
            best_subset = k_subset
            best_results = results

        print 'k: %2d, bic: %.3f, R^2: %.3f, subset: %s, ' % (size_k, results.bic, results.rsquared, str(k_subset))

    print 'Best subset by exhaustive search: %s, bic: %.3f, R^2: %.3f' % (
        str(best_subset), best_results.bic, best_results.rsquared)


def main():
    dataset_1_data = load_dataset_1()

    # heatmap_prob_1a(dataset_1_data)
    exhaustive_search_prob_1b(dataset_1_data)


if __name__ == '__main__':
    main()
