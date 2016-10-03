import pandas as pd

from categorical import convert_categorical_columns


def load_expanded_df():
    df = pd.read_csv('datasets/dataset_4_expanded.txt')
    return df


# prob 4a

def get_expanded_df():
    df = pd.read_csv('datasets/dataset_4.txt', dtype={'NOEXCH': object})
    expanded_df = convert_categorical_columns(df)
    return expanded_df


def fit_regression_model_prob_4a():
    # df = get_expanded_df()
    df = load_expanded_df()

    y_column_name = 'TARGET_D'
    y = df[y_column_name]
    x_df = df.drop(y_column_name, axis=1)

    print df.shape


def main():
    fit_regression_model_prob_4a()


if __name__ == '__main__':
    main()
