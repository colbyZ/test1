from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    df = pd.read_excel('table01.xls')

    row_offsets = range(2, 3)
    column_pairs = [
        (4, 'Reported registered'),
        (10, 'Reported voted'),
    ]

    tuples_dict = OrderedDict()
    for gender_index in range(3):
        start_row = 6 + 76 * gender_index
        gender_label = df.loc[start_row][0]
        for row_offset in row_offsets:
            row = df.loc[start_row + row_offset]
            row_label = row[0]
            for column_pair in column_pairs:
                column_index = column_pair[0]
                column_label = column_pair[1]
                value = row[column_index]
                row_key = (gender_label, row_label)
                column_dict = tuples_dict.get(column_label, OrderedDict())
                if not column_dict:
                    tuples_dict[column_label] = column_dict
                column_dict[row_key] = value

    voting_df = pd.DataFrame(tuples_dict)
    print voting_df

    df2 = voting_df['Reported registered']
    print df2.values

    ind = np.arange(3)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(ind, df2, width=0.35)
    plt.show()


if __name__ == '__main__':
    main()
