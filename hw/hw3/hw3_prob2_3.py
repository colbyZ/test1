from collections import Counter
from collections import namedtuple
from itertools import izip

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression as Lin_Reg

DatasetData = namedtuple('DatasetData', ['x', 'y'])
TaxicabData = namedtuple('TaxicabData', ['x', 'y'])


def save_counter_data():
    # nrows = 10 * 1000
    nrows = None

    xs, ys = get_taxicab_data(nrows)

    x_column = np.vstack(xs)
    y_column = np.vstack(ys)
    table = np.hstack((x_column, y_column))

    df = pd.DataFrame(table)
    df.columns = ['x', 'y']
    df.to_csv('tripdata_pickup_counts.csv', index=False)


def load_taxicab_data():
    data = np.loadtxt('tripdata_pickup_counts.csv', delimiter=',', skiprows=1)

    y = data[:, -1]
    x = data[:, 0]

    x = normalize(x)
    # y = normalize(y)

    return x, y


def loadtxt(file_name):
    return np.loadtxt('datasets/%s' % file_name, delimiter=',', skiprows=1)


def split_y_x(data):
    y = data[:, -1]
    x = data[:, :-1]
    return y, x


def read_dataset_data(filename):
    # Load train set
    data = loadtxt(filename)

    y, x = split_y_x(data)

    return DatasetData(x, y)


# polynomial regression fit

def polynomial_regression_fit(x_train, y_train, degree_of_the_polynomial):
    x_column = np.vstack(x_train)
    poly_x = x_column
    for exponent in range(2, degree_of_the_polynomial + 1):
        new_column = np.power(x_column, exponent)
        poly_x = np.hstack((poly_x, new_column))

    linear_regression = Lin_Reg()
    linear_regression.fit(poly_x, y_train)

    return linear_regression.coef_, linear_regression.intercept_


# polynomial regression predict

def calculate_polynomial_value(coefs, intercept, x):
    poly_sum = intercept
    for i, coef in enumerate(coefs):
        poly_sum += coef * pow(x, i + 1)
    return poly_sum


def polynomial_regression_predict(coefs, intercept, degree_of_the_polynomial, x_test):
    return [calculate_polynomial_value(coefs, intercept, x) for x in x_test]


# polynomial regression score

def polynomial_regression_score(y_predicted, y_test):
    rss = 0.0
    tss = 0.0
    y_mean = np.mean(y_test)
    for predicted_value, actual_value in izip(y_predicted, y_test):
        rss += (actual_value - predicted_value) ** 2
        tss += (actual_value - y_mean) ** 2

    r_squared = 1.0 - rss / tss
    return r_squared, rss


def read_dataset3_data():
    data = loadtxt('dataset_3.txt')

    y = data[:, -1]
    x = data[:, 0]

    return DatasetData(x, y)


def fit_and_visualize_prob_2a():
    x, y = dataset_3_data

    degrees = [3, 5, 10, 25]

    degrees_len = len(degrees)
    _, axes = plt.subplots(degrees_len, 1, figsize=(8, 5 * degrees_len))

    xs = np.linspace(0.01, 0.99)
    for i, degree in enumerate(degrees):
        ax = axes[i]
        coefs, intercept = polynomial_regression_fit(x, y, degree)
        ax.scatter(x, y, color='blue')

        ax.plot(xs, polynomial_regression_predict(coefs, intercept, degree, xs), color='red')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('degree of the polynomial: %d' % degree)

    plt.show()


def evaluate_polynomial_regression_fit(coefs, intercept, x_test, y_test, degree):
    y_predicted = polynomial_regression_predict(coefs, intercept, degree, x_test)
    return polynomial_regression_score(y_predicted, y_test)


def train_test_split_by_index(data, index):
    train = data[:index]
    test = data[index:]
    return train, test


def plot_r_sq(ax, x_train, x_test, y_train, y_test, max_degree=15):
    r_sq_train_list = []
    r_sq_test_list = []
    plot_degrees = []

    degrees = range(1, max_degree + 1)
    for degree in degrees:
        coefs, intercept = polynomial_regression_fit(x_train, y_train, degree)
        r_sq_train, _ = evaluate_polynomial_regression_fit(coefs, intercept, x_train, y_train, degree)
        r_sq_test, _ = evaluate_polynomial_regression_fit(coefs, intercept, x_test, y_test, degree)
        print 'degree: %2d, train R^2: %.5f, test R^2: %.5f' % (degree, r_sq_train, r_sq_test)

        r_sq_train_list.append(r_sq_train)
        r_sq_test_list.append(r_sq_test)
        plot_degrees.append(degree)

    max_r_sq_test = np.max(r_sq_test_list)
    max_r_sq_test_degree = r_sq_test_list.index(max_r_sq_test) + 1
    print 'max test R^2, degree: %d, value: %.5f' % (max_r_sq_test_degree, max_r_sq_test)

    ax.plot(plot_degrees, r_sq_train_list, label='train')
    ax.plot(plot_degrees, r_sq_test_list, label='test')
    ax.legend(loc='lower right')
    ax.set_xlabel('degree of the polynomial')
    ax.set_ylabel('$R^2$')
    ax.set_title('$R^2$ for the training and test sets as a function of the degree')


def compare_errors_prob_2b():
    x, y = dataset_3_data
    mid_index = len(x) / 2

    x_train, x_test = train_test_split_by_index(x, mid_index)
    y_train, y_test = train_test_split_by_index(y, mid_index)

    _, ax = plt.subplots(1, 1, figsize=(12, 6))

    plot_r_sq(ax, x_train, x_test, y_train, y_test)

    plt.show()


def compute_aic(n, rss, degree):
    return n * np.log(rss / n) + 2 * degree


def compute_bic(n, rss, degree):
    return n * np.log(rss / n) + np.log(n) * degree


def plot_aic_and_bic(ax, x, y, max_degree=15):
    n = len(x)
    aic_list = []
    bic_list = []
    degrees = range(1, max_degree + 1)
    for degree in degrees:
        coefs, intercept = polynomial_regression_fit(x, y, degree)
        r_sq, rss = evaluate_polynomial_regression_fit(coefs, intercept, x, y, degree)
        aic = compute_aic(n, rss, degree)
        aic_list.append(aic)
        bic = compute_bic(n, rss, degree)
        bic_list.append(bic)
        print 'degree: %2d, AIC: %.1f, BIC: %.1f' % (degree, aic, bic)

    min_aic = np.min(aic_list)
    min_aic_degree = aic_list.index(min_aic) + 1
    print 'min aic, degree: %d, value: %.5f' % (min_aic_degree, min_aic)

    min_bic = np.min(bic_list)
    min_bic_degree = bic_list.index(min_bic) + 1
    print 'min bic, degree: %d, value: %.5f' % (min_bic_degree, min_bic)

    ax.plot(degrees, aic_list, label='AIC')
    ax.plot(degrees, bic_list, label='BIC')
    ax.legend(loc='upper right')
    ax.set_xlabel('degree of the polynomial')
    ax.set_title('AIC and BIC as functions of the degree')


def compute_aic_and_bic():
    x, y = dataset_3_data

    _, ax = plt.subplots(1, 1, figsize=(12, 6))

    plot_aic_and_bic(ax, x, y)

    plt.show()


def normalize(xs):
    xs = np.array(xs, dtype=float)
    min_x = xs.min()
    max_x = xs.max()
    return [(x - min_x) / max_x for x in xs]


def get_taxicab_data(nrows=None):
    df = pd.read_csv('green_tripdata_2015-01.csv', header=0, index_col=False, usecols=['lpep_pickup_datetime'],
                     parse_dates=['lpep_pickup_datetime'], nrows=nrows)

    day_minute_list = []
    for index, row in df.iterrows():
        datetime = row[0]
        minute = datetime.minute
        hour = datetime.hour
        day_minute = minute + 60 * hour
        day_minute_list.append(day_minute)

    counter = Counter(day_minute_list)
    xs, ys = zip(*counter.items())

    return TaxicabData(normalize(xs), ys)


def taxicab_plot_data():
    xs, ys = taxicab_data

    _, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(xs, ys)
    ax.set_xlabel('"normalized" time of the day (in minutes divided by 1440)')
    ax.set_ylabel('number of pickups')

    plt.tight_layout()
    plt.show()


def taxicab_density_estimation():
    xs, ys = taxicab_data

    np.random.seed(1090)

    degrees = [1, 5, 10, 20, 40]
    len_degrees = len(degrees)
    num_axes = 2 + len_degrees
    _, axes = plt.subplots(num_axes, 1, figsize=(10, 5 * num_axes))

    x_train, x_test, y_train, y_test = train_test_split(xs, ys, train_size=0.7)

    max_degree = 40
    plot_r_sq(axes[0], x_train, x_test, y_train, y_test, max_degree=max_degree)
    plot_aic_and_bic(axes[1], xs, ys, max_degree)

    lin_xs = np.linspace(0.0, 1.0)
    for i, degree in enumerate(degrees):
        ax = axes[2 + i]
        coefs, intercept = polynomial_regression_fit(xs, ys, degree)
        print 'degree: %d, intercept: %.1f' % (degree, intercept)
        for ci, coef in enumerate(coefs):
            print 'coef %d: %.1f' % (ci + 1, coef)
        ax.scatter(xs, ys, color='blue')

        ax.plot(lin_xs, polynomial_regression_predict(coefs, intercept, degree, lin_xs), color='red')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('degree of the polynomial: %d' % degree)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # dataset_3_data = read_dataset3_data()
    # fit_and_visualize_prob_2a()
    # compare_errors_prob_2b()
    # compute_aic_and_bic()

    # save_counter_data()

    # taxicab_data = get_taxicab_data()
    taxicab_data = load_taxicab_data()

    # print type(counter_xs.dtype)

    taxicab_plot_data()
