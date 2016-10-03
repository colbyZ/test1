import pandas as pd


def fit_regression_model_prob_4a():
    df = pd.read_csv('datasets/dataset_4.txt', dtype={'NOEXCH': object})

    y_column_name = 'TARGET_D'
    y = df[y_column_name]
    x_df = df.drop(y_column_name, axis=1)

    print x_df.shape, y.shape


def main():
    fit_regression_model_prob_4a()


if __name__ == '__main__':
    main()
