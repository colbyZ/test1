from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from bs4 import BeautifulSoup
from statsmodels.sandbox.regression.predstd import wls_prediction_std

ColumnInfo = namedtuple('ColumnInfo', ['values', 'name', 'units'])
Interval = namedtuple('Interval', ['lower', 'upper'])


def get_row(row_tag, cell_tag_name):
    return [tag.get_text() for tag in row_tag.find_all(cell_tag_name)]


def get_data_frame():
    soup = BeautifulSoup(open("index.html"), 'lxml')
    table_tag = soup.find('table')
    tr_tag_list = table_tag.find_all('tr')

    header_tag = tr_tag_list[0]
    row_tags = tr_tag_list[1:]

    header_values = get_row(header_tag, 'th')

    data_array = np.empty((0, 4))
    for row in row_tags:
        row_values = pd.to_numeric(get_row(row, 'td'))
        data_array = np.vstack((data_array, row_values))

    return pd.DataFrame(data_array, columns=header_values)


def get_label_text(column):
    return '%s (%s)' % (column.name, column.units)


def show_scatter_plot(ax, x_column, y_column):
    ax.scatter(x_column.values, y_column.values)

    ax.set_xlabel(get_label_text(x_column))
    ax.set_ylabel(get_label_text(y_column))

    ax.set_title('%s vs %s (scatter plot)' % (x_column.name, y_column.name))


def show_scatter_plots(df):
    times = df['Time']
    xs = df['X-Coord']
    ys = df['Y-Coord']
    zs = df['Z-Coord']

    x_column = ColumnInfo(xs, 'x', 'km')
    y_column = ColumnInfo(ys, 'y', 'km')
    z_column = ColumnInfo(zs, 'z', 'km')
    time_column = ColumnInfo(times, 'time', 'secs')

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    show_scatter_plot(axes[0][0], z_column, x_column)
    show_scatter_plot(axes[0][1], z_column, y_column)
    show_scatter_plot(axes[0][2], z_column, time_column)

    show_scatter_plot(axes[1][0], x_column, y_column)
    show_scatter_plot(axes[1][1], time_column, x_column)
    show_scatter_plot(axes[1][2], time_column, y_column)

    plt.tight_layout()
    plt.show()


def get_polynomials(xs, degree):
    new_xs = xs
    for d in xrange(2, degree + 1):
        new_xs = np.hstack((new_xs, xs ** d))
    return new_xs


def find_interval(xs, ys, description_text):
    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)

    best_bic = float('inf')
    best_results = None
    best_degree = None

    for degree in xrange(1, 9):
        poly_xs = get_polynomials(xs, degree)

        model = sm.OLS(ys, sm.add_constant(poly_xs))
        results = model.fit()

        aic = results.aic
        bic = results.bic
        rsquared_adj = results.rsquared_adj

        print '%s, degree: %d, aic: %.1f, bic: %.1f, R^2 adj: %.4f' % (description_text, degree, aic, bic, rsquared_adj)

        if bic < best_bic:
            best_bic = bic
            best_results = results
            best_degree = degree

    print '%s, best, degree: %d, bic: %.1f' % (description_text, best_degree, best_bic)

    zeros = np.append([1.0], np.zeros(best_degree))
    predictions = best_results.predict(zeros)

    prstd, iv_l, iv_u = wls_prediction_std(best_results, zeros, alpha=1.0 - np.sqrt(0.9))

    prediction = predictions[0]
    lower_limit = iv_l[0]
    upper_limit = iv_u[0]
    print '%s, prediction: %.2f, 94.9%% interval: [%.2f-%.2f]\n' % (
        description_text, prediction, lower_limit, upper_limit)

    return Interval(lower_limit, upper_limit)


def inside(interval, value):
    return interval.lower <= value <= interval.upper


def count_residents(x_interval, y_interval):
    df = pd.read_csv('pop_data.csv').fillna(0)

    num_residents = sum(row['residents']
                        for index, row in df.iterrows()
                        if inside(x_interval, row['x']) and inside(y_interval, row['y']))

    print 'residents within the region: %d' % num_residents


def fit_all(df):
    xs = df['X-Coord']
    ys = df['Y-Coord']
    zs = df['Z-Coord']

    x_interval = find_interval(zs, xs, 'x')
    y_interval = find_interval(zs, ys, 'y')

    count_residents(x_interval, y_interval)


def asteroid_analysis():
    df = get_data_frame()

    # print 'dataframe:\n%s' % df

    fit_all(df)

    # show_scatter_plots(df)


def main():
    asteroid_analysis()


if __name__ == '__main__':
    main()
