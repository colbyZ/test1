import pandas as pd


def split(df, split_index):
    train = df[:split_index]
    test = df[split_index:]
    return train, test


def is_categorical(column):
    return column.dtype == object or len(column.unique()) < 8


def convert_categorical_columns(x_df):
    expanded_x_df = x_df.copy()
    for column_name in x_df.columns:
        column = x_df[column_name]
        if is_categorical(column):
            dummies_df = pd.get_dummies(column, prefix=column_name)

            expanded_x_df.drop(column_name, axis=1, inplace=True)
            expanded_x_df = expanded_x_df.join(dummies_df)

    return expanded_x_df
