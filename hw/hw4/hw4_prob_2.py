import pandas as pd


def has_limited_unique_values(column):
    unique_values = set(column)
    num_unique_values = len(unique_values)
    return num_unique_values <= 8


def is_categorical(column):
    return column.dtype == object or has_limited_unique_values(column)


def encode_categorical_variables_prob_2a():
    df = pd.read_csv('datasets/dataset_2.txt')

    dummy_df = df.copy()
    for column_name in df.columns:
        column = df[column_name]
        if is_categorical(column):
            dummies_df = pd.get_dummies(column, prefix=column_name)

            dummy_df = dummy_df.drop(column_name, axis=1)
            dummy_df = pd.concat([dummy_df, dummies_df], axis=1)

    print dummy_df.columns
    print len(dummy_df.columns)


def main():
    encode_categorical_variables_prob_2a()


if __name__ == '__main__':
    main()
