import matplotlib.pyplot as plt
import pandas as pd


def split(df, split_index):
    train = df[:split_index]
    test = df[split_index:]
    return train, test


def is_categorical(column):
    return column.dtype == object or len(column.unique()) < 8


def convert_categorical_columns(df):
    expanded_df = df.copy()
    for column_name in df.columns:
        column = df[column_name]
        if is_categorical(column):
            dummies_df = pd.get_dummies(column, prefix=column_name)

            expanded_df.drop(column_name, axis=1, inplace=True)
            expanded_df = expanded_df.join(dummies_df)

    return expanded_df


def plot_train_test_scores(alphas, test_score_list, train_score_list, ylim, legend_loc):
    # Plot
    plt.plot(alphas, train_score_list, label='train')
    plt.plot(alphas, test_score_list, label='test')

    plt.xscale('log')
    plt.xlabel('$\lambda$')
    plt.ylabel('$R^2$')
    plt.title('train and test $R^2$ scores as a function of the regularization parameter')
    plt.ylim(ylim)
    plt.legend(loc=legend_loc)

    plt.show()
