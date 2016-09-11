import pandas as pd


def main():
    df = pd.read_excel('table01.xls')

    print df.columns.values[0], '\n'

    for j in range(3):
        start = 6 + 76 * j
        list1 = [df.loc[i][0] for i in range(start, start + 75)]
        print list1
    print

    column_headers = df.loc[2:4]
    print column_headers


if __name__ == '__main__':
    main()
