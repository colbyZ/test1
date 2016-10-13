import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


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


def asteroid():
    df = get_data_frame()

    print 'dataframe:\n%s' % df


def main():
    asteroid()


if __name__ == '__main__':
    main()
