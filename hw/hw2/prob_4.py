import operator
from itertools import izip

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def show_histogram(zs, title, mean=0.0, std=1.0, show_normal=True):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.hist(zs, 400, normed=True)

    if show_normal:
        # plot the pdf of the normal distribution
        xs = np.linspace(mean - 4 * std, mean + 4 * std, num=100)
        ax.plot(xs, norm.pdf(xs, mean, std))

    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def add_variables(variable_list):
    return (sum(elements) for elements in izip(*variable_list))


def create_k_uniform_variables(n, k):
    return (np.random.uniform(0.0, 1.0, n) for _ in range(k))


def print_stats(zs):
    print 'mean: %.3f, std: %.3f' % (np.mean(zs), np.std(zs))


def add_uniform_variables(k):
    sample_size = 10000000 / k
    variable_list = create_k_uniform_variables(sample_size, k)
    zs = list(add_variables(variable_list))
    print_stats(zs)
    show_histogram(zs, 'sum of %d uniformly distributed variables' % k, k / 2.0, np.sqrt(k / 12.0))


def create_k_normal_variables(n, k):
    return (np.random.normal(0.0, 1.0, n) for _ in range(k))


def add_normal_variables(k):
    sample_size = 10000000 / k
    variable_list = create_k_normal_variables(sample_size, k)
    zs = list(add_variables(variable_list))
    print_stats(zs)
    show_histogram(zs, 'sum of %d normally distributed variables' % k, 0, np.sqrt(2))


def multiply_variables(variable_list):
    return (reduce(operator.mul, elements) for elements in izip(*variable_list))


def multiply_normal_variables(k):
    sample_size = 10000000 / k
    variable_list = create_k_normal_variables(sample_size, k)
    zs = list(multiply_variables(variable_list))
    print_stats(zs)
    show_histogram(zs, 'product of %d normally distributed variables' % k, show_normal=False)


if __name__ == '__main__':
    np.random.seed(1090)

    # add_uniform_variables(2)
    # add_uniform_variables(7)
    add_normal_variables(2)
    multiply_normal_variables(2)
