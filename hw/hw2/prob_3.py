from itertools import izip

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as Lin_Reg


def reshape(df, column_name):
    return df[column_name].reshape((len(df), 1))


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


def visualize_fit():
    df = read_dataset(1)

    x_values = df['x'].values
    y_values = df['y'].values

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.scatter(x_values, y_values)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('different linear fits')

    x_train = reshape(df, 'x')
    y_train = reshape(df, 'y')

    lin_reg = Lin_Reg()
    lin_reg.fit(x_train, y_train)

    x = np.arange(-0.1, 2.0, step=1.2)
    ax.plot(x, 0.4 * x + 0.2, label='slope = 0.4, intercept = 0.2')
    ax.plot(x, 0.4 * x + 4, label='slope = 0.4, intercept = 4')
    ax.plot(x, lin_reg.coef_[0][0] * x + lin_reg.intercept_[0], label='linear regression model')
    ax.legend(loc='upper left')

    plt.show()


def compute_residuals(x_values, y_values, slope, intercept):
    residuals = []
    for x, y in izip(x_values, y_values):
        residuals.append(y - (x * slope + intercept))
    return residuals


def residual_plots():
    df = read_dataset(1)

    x_values = df['x'].values
    y_values = df['y'].values

    x_train = reshape(df, 'x')
    y_train = reshape(df, 'y')

    lin_reg = Lin_Reg()
    lin_reg.fit(x_train, y_train)

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    fit_list = [
        (0.4, 0.2, 'slope = 0.4, intercept = 0.2'),
        (0.4, 4, 'slope = 0.4, intercept = 4'),
        (lin_reg.coef_[0][0], lin_reg.intercept_[0], 'linear regression model'),
    ]
    for i, fit in enumerate(fit_list):
        ax1 = axes[i][0]
        residuals = compute_residuals(x_values, y_values, fit[0], fit[1])
        ax1.scatter(x_values, residuals)
        ax1.set_xlabel('x')
        ax1.set_ylabel('residuals')
        ax1.set_title(fit[2])
        ax1.plot((-0.1, 1.1), (0, 0))

    plt.show()


if __name__ == '__main__':
    # read_and_visualize_dataset()
    # visualize_fit()
    residual_plots()
