import pandas as pd


# prob 2a

def is_categorical(column):
    return column.dtype == object or len(column.unique()) < 8


def encode_categorical_variables_prob_2a():
    df = pd.read_csv('datasets/dataset_2.txt')

    expanded_df = df.copy()
    for column_name in df.columns:
        column = df[column_name]
        if is_categorical(column):
            dummies_df = pd.get_dummies(column, prefix=column_name)

            expanded_df = expanded_df.drop(column_name, axis=1)
            expanded_df = pd.concat([expanded_df, dummies_df], axis=1)

    return expanded_df


def print_expanded_df_prob_2a(df):
    print '%d columns:' % len(df.columns)
    print df.columns


# prob 2b

def linear_regression_prob_2b(df):
    print df.head()


def main():
    expanded_df = encode_categorical_variables_prob_2a()

    # print_expanded_df_prob_2a(expanded_df)
    linear_regression_prob_2b(expanded_df)


if __name__ == '__main__':
    main()
