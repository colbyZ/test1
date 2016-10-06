from itertools import izip

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Lasso as Lasso_Reg

from hw4_common import convert_categorical_columns
from hw4_common import plot_train_test_scores


def load_expanded_df():
    return pd.read_csv('datasets/dataset_4_expanded.txt')


# prob 4a

def get_expanded_df():
    df = pd.read_csv('datasets/dataset_4.txt', dtype={'NOEXCH': object})
    expanded_df = convert_categorical_columns(df)
    # expanded_df.to_csv('dataset_4_expanded.txt', index=False)
    return expanded_df


def print_profits(name, cost, revenue):
    print '%s, cost: %d, revenue: %d, profit: %d' % (name, cost, revenue, revenue - cost)


def get_best_regression(x, y):
    random_state = 1090
    # random_state = None

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state)

    alphas = np.logspace(-4.0, -1.0, num=20)

    best_test_score = -float('inf')
    best_regression = None
    train_score_list = []
    test_score_list = []

    for alpha in alphas:
        regression = Lasso_Reg(alpha=alpha, normalize=True, tol=5e-3)
        regression.fit(x_train, y_train)
        train_score = regression.score(x_train, y_train)
        test_score = regression.score(x_test, y_test)

        if test_score > best_test_score:
            best_test_score = test_score
            best_regression = regression

        num_non_zero_coefs = sum(abs(coef) > 0.0 for coef in regression.coef_)

        train_score_list.append(train_score)
        test_score_list.append(test_score)

        print 'alpha: %.2e, test R^2: % .4f, train R^2: %.4f, num_non_zero_coefs: %4d' % (
            alpha, test_score, train_score, num_non_zero_coefs)

    print 'best test score: %.4f' % best_test_score

    return best_regression, alphas, test_score_list, train_score_list


def fit_regression_model_prob_4():
    df = get_expanded_df()
    # df = load_expanded_df()

    y_column_name = 'TARGET_D'
    y = df[y_column_name]
    x = df.drop(y_column_name, axis=1)

    random_state = 1091
    # random_state = None

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state)

    best_regression, alphas, test_score_list, train_score_list = get_best_regression(x_train, y_train)

    print 'best regression: %s' % best_regression

    predicted_ys = best_regression.predict(x_test)

    blanket_cost = 0.0
    blanket_revenue = 0.0

    model_cost = 0.0
    model_revenue = 0.0

    for predicted_y, y in izip(predicted_ys, y_test):
        blanket_cost += 7.0
        blanket_revenue += y

        if predicted_y > 7.0:
            model_cost += 7.0
            model_revenue += y

    print_profits('blanket', blanket_cost, blanket_revenue)
    print_profits('model', model_cost, model_revenue)

    plot_train_test_scores(alphas, test_score_list, train_score_list, [-0.6, 0.7], 'upper right')


def main():
    fit_regression_model_prob_4()


if __name__ == '__main__':
    main()
