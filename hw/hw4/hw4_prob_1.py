import numpy as np
import pandas as pd


def prob_1a():
    # Load data
    data = np.loadtxt('datasets/dataset_1.txt', delimiter=',', skiprows=1)

    # Split predictors and response
    x = data[:, :-1]
    y = data[:, -1]

    df = pd.DataFrame(data)
    print repr(df.head())


def main():
    prob_1a()


if __name__ == '__main__':
    main()
