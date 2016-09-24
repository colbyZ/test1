import pandas as pd


def taxicab_density_estimation():
    nrows = 10
    df = pd.read_csv('green_tripdata_2015-01.csv', header=0, index_col=False, usecols=['lpep_pickup_datetime'],
                     parse_dates=['lpep_pickup_datetime'], nrows=nrows)
    print df.ix[0]


if __name__ == '__main__':
    taxicab_density_estimation()
