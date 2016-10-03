import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Lasso as Lasso_Reg

from hw4_categorical import convert_categorical_columns


def load_expanded_df():
    df = pd.read_csv('datasets/dataset_4_expanded.txt')
    return df


# prob 4a

def get_expanded_df():
    df = pd.read_csv('datasets/dataset_4.txt', dtype={'NOEXCH': object})
    expanded_df = convert_categorical_columns(df)
    # expanded_df.to_csv('dataset_4_expanded.txt')
    return expanded_df


def fit_regression_model_prob_4a():
    # df = get_expanded_df()
    df = load_expanded_df()

    y_column_name = 'TARGET_D'
    y = df[y_column_name]
    x = df.drop(y_column_name, axis=1)

    random_state = 1090
    # random_state = None

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state)

    alphas = [10.0 ** i for i in np.arange(-4.0, -1.1, step=0.1)]

    best_test_score = -float('inf')
    best_regression = None

    for alpha in alphas:
        regression = Lasso_Reg(alpha=alpha, normalize=True, tol=2e-3)
        regression.fit(x_train, y_train)
        train_score = regression.score(x_train, y_train)
        test_score = regression.score(x_test, y_test)

        if test_score > best_test_score:
            best_test_score = test_score
            best_regression = regression

        num_non_zero_coefs = sum(abs(coef) > 0.0 for coef in regression.coef_)

        print 'alpha: %.2e, test R^2: % .3f, train R^2: %.3f, num_non_zero_coefs: %4d' % (
            alpha, test_score, train_score, num_non_zero_coefs)

    print 'best test score: %.3f' % best_test_score
    print 'best regression: %s' % str(best_regression)


def main():
    fit_regression_model_prob_4a()


if __name__ == '__main__':
    main()
