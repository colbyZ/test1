import pandas as pd


def is_categorical(column):
    return column.dtype == object or len(column.unique()) < 8


def convert_categorical_columns(x_df):
    expanded_x_df = x_df.copy()
    for column_name in x_df.columns:
        column = x_df[column_name]
        if is_categorical(column):
            dummies_df = pd.get_dummies(column, prefix=column_name)

            expanded_x_df = expanded_x_df.drop(column_name, axis=1)
            expanded_x_df = pd.concat([expanded_x_df, dummies_df], axis=1)

    return expanded_x_df
