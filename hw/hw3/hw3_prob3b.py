from collections import Counter

import pandas as pd


def taxicab_density_estimation():
    nrows = 100 * 1000
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
    time_of_the_day_list, num_pickups_list = zip(*counter.items())


if __name__ == '__main__':
    taxicab_density_estimation()
