from itertools import izip

import matplotlib.pyplot as plt
import numpy as np


def create_uniform_sample(n):
    xs = np.random.uniform(0.0, 1.0, n)
    return xs


def add_two_uniformly_distributed_variables():
    np.random.seed(1090)

    n = 2000000
    xs = create_uniform_sample(n)
    ys = create_uniform_sample(n)
    zs = []
    for x, y in izip(xs, ys):
        zs.append(x + y)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.hist(zs, 400, normed=True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    add_two_uniformly_distributed_variables()
