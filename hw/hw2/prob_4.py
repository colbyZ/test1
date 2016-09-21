import matplotlib.pyplot as plt
import numpy as np


def create_uniform_sample(n):
    xs = np.random.uniform(0.0, 1.0, n)
    return xs


def add_variables(variable_list):
    zs = []
    for i, vs in enumerate(variable_list):
        if i == 0:
            for v in vs:
                zs.append(v)
        else:
            for j, v in enumerate(vs):
                zs[j] += v
    return zs


def show_histogram(zs):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.hist(zs, 400, normed=True)

    plt.tight_layout()
    plt.show()


def add_two_uniformly_distributed_variables():
    np.random.seed(1090)

    n = 2000000
    xs = create_uniform_sample(n)
    ys = create_uniform_sample(n)
    zs = add_variables([xs, ys])

    show_histogram(zs)


def add_three_uniformly_distributed_variables():
    np.random.seed(1090)

    n = 2000000
    zs = add_variables([
        (create_uniform_sample(n)),
        (create_uniform_sample(n)),
        (create_uniform_sample(n)),
    ])

    show_histogram(zs)


if __name__ == '__main__':
    # add_two_uniformly_distributed_variables()
    add_three_uniformly_distributed_variables()
