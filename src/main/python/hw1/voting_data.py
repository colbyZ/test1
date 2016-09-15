import matplotlib.pyplot as plt
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


# main2()

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


def clean_column_value(s):
    return '' if s.startswith('Unnamed: ') else s


def concat_column_values(v1, v2):
    return v1 + ' - ' + v2 if v2 else v1


def d(*args):
    return ' - '.join(value for value in args if value)


def get_column_label_list(values):
    str_list = []
    for v in values:
        v0 = v[0]
        v1 = clean_column_value(v[1])
        v2 = clean_column_value(v[2])
        s = concat_column_values(v0, v1)
        s = concat_column_values(s, v2)
        str_list.append(s)
    return str_list


def get_indices_to_drop():
    ind_list = []
    for j in range(3):
        st = j * 76
        ind_list.extend(range(st + 0, st + 11))
        if j < 2:
            ind_list.append(st + 75)
    return ind_list


def get_age_list():
    ind_list = list(range(18, 81))
    ind_list.append(85)
    return ind_list


def read_excel():
    df = pd.read_excel('table01.xls', skiprows=3, skip_footer=5, header=[0, 1, 2], index_col=0)

    # create simple column names instead of multi-level ones
    df.columns = get_column_label_list(df.columns.values)

    # we drop the columns that we are not interested in
    df = df.drop(df.columns[[4, 5, 6, 7, 10, 11, 12, 13]], axis=1)

    # create the gender column so that later we can create an index with the gender component
    gender_label_list = get_gender_label_list(df.index.values)
    df.insert(0, 'gender', gender_label_list)

    # create a separate 'total' dataframe
    total_df = df.iloc[[2, 78, 154]]
    total_df = total_df.set_index(['gender'])

    # drop all rows except for the individual age ones
    indices_to_drop = get_indices_to_drop()
    age_df = df.drop(df.index[indices_to_drop])

    # create the age column
    age_list = 3 * get_age_list()
    age_df.insert(1, 'age', age_list)

    # create ('gender', 'age') multi index for the main dataframe
    age_df = age_df.set_index(['gender', 'age'])

    return age_df, total_df


def show_plot(title, plot_info_1, plot_info_2, ylabel):
    plt.plot(plot_info_1[0], label=plot_info_1[1])
    plt.plot(plot_info_2[0], label=plot_info_2[1])
    plt.xlabel('Age')
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()


def show_percent_plot(title, plot_info_1, plot_info_2):
    show_plot(title, plot_info_1, plot_info_2, 'Percent')


def show_number_plot(title, plot_info_1, plot_info_2):
    show_plot(title, plot_info_1, plot_info_2, 'Population')


def main2():
    age_df, total_df = read_excel()
    # print 'total dataframe:\n%s\n' % total_df
    # print 'age dataframe:\n%s\n' % age_df.head()
    # print 'age dataframe (female):\n%s\n' % age_df.loc['Female'].head()

    print age_df.columns

    # print total_df.loc[:][['US Citizen - Reported registered - Percent', 'US Citizen - Reported voted - Percent']]

    show_number_plot(
        'Reported voted by gender',
        (age_df.loc['Female'][['US Citizen - Reported voted - Number']], 'female'),
        (age_df.loc['Male'][['US Citizen - Reported voted - Number']], 'male'))

    # show_plot(
    #     'Reported voted by gender',
    #     (age_df.loc['Female'][['US Citizen - Reported voted - Percent']], 'female'),
    #     (age_df.loc['Male'][['US Citizen - Reported voted - Percent']], 'male'))


if __name__ == '__main__':
    main2()
