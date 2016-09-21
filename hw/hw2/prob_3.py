from collections import namedtuple
from itertools import izip

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as Lin_Reg

Problem_3a_Data = namedtuple('Problem_3a_Data', ['df', 'lin_reg_fit', 'fit_info_list'])
LinearFit = namedtuple('LinearFit', ['slope', 'intercept'])
FitInfo = namedtuple('FitInfo', ['linear_fit', 'title'])


def read_dataset(dataset_i):
    return pd.read_csv('./dataset/dataset_%d_full.txt' % dataset_i)


def reshape_column(df, column_name):
    return df[column_name].reshape(-1, 1)


def prepare_problem_3a_data():
    df = read_dataset(1)
    x_train = reshape_column(df, 'x')
    y_train = reshape_column(df, 'y')
    lin_reg = Lin_Reg()
    lin_reg.fit(x_train, y_train)
    lin_reg_fit = LinearFit(lin_reg.coef_[0][0], lin_reg.intercept_[0])
    fit_info_list = [
        FitInfo(LinearFit(0.4, 0.2), 'slope = 0.4, intercept = 0.2'),
        FitInfo(LinearFit(0.4, 4.0), 'slope = 0.4, intercept = 4'),
        FitInfo(lin_reg_fit, 'linear regression model'),
    ]
    return Problem_3a_Data(df, lin_reg_fit, fit_info_list)


def read_and_visualize_dataset():
    df = problem_3a_data.df

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
    ax1.scatter(df[['x']].values, df[['y']].values)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Dataset 1 scatter plot')
    plt.show()


def visualize_fit():
    df = problem_3a_data.df
    lin_reg_fit = problem_3a_data.lin_reg_fit

    x_values = df['x']
    y_values = df['y'].values

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.scatter(x_values, y_values)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('different linear fits')

    x = np.arange(-0.1, 2.0, step=1.2)
    ax.plot(x, 0.4 * x + 0.2, label='slope = 0.4, intercept = 0.2')
    ax.plot(x, 0.4 * x + 4, label='slope = 0.4, intercept = 4')
    ax.plot(x, lin_reg_fit.slope * x + lin_reg_fit.intercept, label='linear regression model')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def compute_residuals(x_values, y_values, linear_fit):
    residuals = []
    for x, y in izip(x_values, y_values):
        residuals.append(y - (x * linear_fit.slope + linear_fit.intercept))
    return residuals


def residual_plots():
    df = problem_3a_data.df

    x_values = df['x']
    y_values = df['y']

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    for i, fit_info in enumerate(problem_3a_data.fit_info_list):
        ax1 = axes[i][0]
        residuals = compute_residuals(x_values, y_values, fit_info.linear_fit)
        ax1.scatter(x_values, residuals)
        ax1.set_xlabel('x')
        ax1.set_ylabel('residuals')
        ax1.set_title(fit_info.title)
        ax1.plot((-0.1, 1.1), (0, 0))

        ax2 = axes[i][1]
        ax2.hist(residuals, 30)
        ax2.set_xlabel('residuals')
        ax2.set_title('residual histogram')

    plt.tight_layout()
    plt.show()


def calculate_r_squared_coefs():
    pass


if __name__ == '__main__':
    problem_3a_data = prepare_problem_3a_data()

    # read_and_visualize_dataset()
    visualize_fit()
    # residual_plots()
    # calculate_r_squared_coefs()
