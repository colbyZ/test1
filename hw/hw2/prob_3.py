import matplotlib.pyplot as plt
import pandas as pd


def residual_plots_a():
    dataset_i = 1
    df = pd.read_csv('./dataset/dataset_%d_full.txt' % dataset_i)

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
    ax1.scatter(df[['x']].values, df[['y']].values)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Dataset 1 scatter plot')
    plt.show()


if __name__ == '__main__':
    residual_plots_a()
