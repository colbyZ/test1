from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

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

    show_scatter_plot(axes[0][0], x_column, z_column)
    show_scatter_plot(axes[0][1], y_column, z_column)
    show_scatter_plot(axes[0][2], time_column, z_column)

    show_scatter_plot(axes[1][0], x_column, y_column)
    show_scatter_plot(axes[1][1], time_column, y_column)
    show_scatter_plot(axes[1][2], x_column, time_column)

    plt.tight_layout()
    plt.show()


def asteroid():
    df = get_data_frame()

    print 'dataframe:\n%s' % df

    show_scatter_plots(df)


def main():
    asteroid()


if __name__ == '__main__':
    main()
