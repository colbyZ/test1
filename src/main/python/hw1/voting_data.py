import pandas as pd


def is_value_present(s):
    # it's either a string (True) or nan (False)
    return isinstance(s, basestring)


def get_value(column_headers, index, level):
    current_index = index
    result = ''
    while current_index >= 0:
        header = column_headers.iloc[:, current_index]
        v = header[level]
        if is_value_present(v):
            result = v
            break
        current_index -= 1

    return result


def get_column_header_info(column_headers, index):
    header = column_headers.iloc[:, index]
    h3 = get_value(column_headers, index, 3)
    h2 = get_value(column_headers, index, 2)
    column_type = header[4] if is_value_present(header[4]) else ''
    return h2, h3, column_type


def main():
    df = pd.read_excel('table01.xls')

    columns = df.columns
    print "The first column:\n'%s'\n" % columns.values[0]

    print "Column A:"
    for j in range(3):
        start = 6 + 76 * j
        list1 = [df.loc[i][0] for i in range(start, start + 75)]
        print list1
    print

    print "Column headers in rows 4 through 6:"
    column_headers = df.loc[2:4]
    column_header_info_list = [get_column_header_info(column_headers, index) for index, column in enumerate(columns)]
    for info in column_header_info_list:
        print info


def get_gender_label_list(index_values):
    current_label = ''
    gender_labels = ['Both sexes', 'Male', 'Female', '']
    index = 0
    gender_label_list = []
    for v in index_values:
        if v == gender_labels[index]:
            current_label = gender_labels[index]
            index += 1
        gender_label_list.append(current_label)
    return gender_label_list


def main2():
    df = pd.read_excel('table01.xls', skiprows=3, skip_footer=5, header=[0, 1, 2], index_col=0)
    # print df.index
    gender_label_list = get_gender_label_list(df.index.values)
    print gender_label_list
    df.insert(0, 'gender', gender_label_list)
    # df.set_index(['gender', 'index'], inplace=True)
    print df


if __name__ == '__main__':
    main2()
