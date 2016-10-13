from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.cross_validation import train_test_split, cross_val_score, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

ColumnInfo = namedtuple('ColumnInfo', ['values', 'name', 'units'])


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


def fit(xs, ys):
    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(xs, ys)

    print x_train.shape, x_test.shape, y_train.shape, y_test.shape

    best_score = -float('inf')
    best_degree = None
    for degree in xrange(1, 10):
        model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                          ('linear', LinearRegression())])

        cv = ShuffleSplit(n=len(xs), n_iter=3000)

        score = cross_val_score(model, xs, ys, cv=cv).mean()

        print 'degree: %2d, score: %.6f' % (degree, score)

        if score > best_score:
            best_score = score
            best_degree = degree

    print 'best, degree: %d, score: %.6f' % (best_degree, best_score)


def fit_all(df):
    xs = df['X-Coord']
    ys = df['Y-Coord']
    zs = df['Z-Coord']

    fit(xs, zs)


def asteroid_analysis():
    df = get_data_frame()

    print 'dataframe:\n%s' % df

    fit_all(df)

    # show_scatter_plots(df)


def main():
    asteroid_analysis()


if __name__ == '__main__':
    main()
