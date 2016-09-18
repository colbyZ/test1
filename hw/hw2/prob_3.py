import matplotlib.pyplot as plt
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

    fig, axes = plt.subplots(1, 3, figsize=(25, 5))
    x_values = df[['x']].values
    y_values = df[['y']].values

    titles = ['slope = 0.4, intercept = 0.2', 'slope = 0.4, intercept = 4', 'linear regression model']
    for i, title in enumerate(titles):
        ax = axes[i]
        ax.scatter(x_values, y_values)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)

    plt.show()


if __name__ == '__main__':
    # read_and_visualize_dataset()
    residual_plots()
