import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_dataset(dataset_i):
    return pd.read_csv('./dataset/dataset_%d_full.txt' % dataset_i)


def read_and_visualize_dataset():
    df = read_dataset(1)

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
    ax1.scatter(df[['x']].values, df[['y']].values)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Dataset 1 scatter plot')
    plt.show()


def residual_plots():
    df = read_dataset(1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    x_values = df[['x']].values
    y_values = df[['y']].values

    # titles = ['slope = 0.4, intercept = 0.2', 'slope = 0.4, intercept = 4', 'linear regression model']
    ax.scatter(x_values, y_values)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Dataset 1')

    x = np.arange(-0.1, 2.0, step=1.2)
    plt.plot(x, 0.4 * x + 0.2, label='slope = 0.4, intercept = 0.2')
    plt.plot(x, 0.4 * x + 4, label='slope = 0.4, intercept = 4')
    plt.legend(loc='upper left')

    plt.show()


if __name__ == '__main__':
    # read_and_visualize_dataset()
    residual_plots()
