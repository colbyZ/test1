import math
import pandas as pd
from sklearn.cross_validation import train_test_split


def compute_means(df):
    marker_columns = [1, 2]
    subtype = df['subtype']

    column_names = df[marker_columns].columns.values
    means = pd.DataFrame(columns=column_names)

    for st in range(4):
        means.loc[st] = df[subtype == st][marker_columns].mean()

    return means


def distance(marker1, marker2, m1_mean, m2_mean):
    sq1 = math.pow(marker1 - m1_mean, 2)
    sq2 = math.pow(marker2 - m2_mean, 2)
    return math.sqrt(sq1 + sq2)


def classify_markers(marker1, marker2, means):
    distances = [distance(marker1, marker2, row['marker_1'], row['marker_2']) for index, row in means.iterrows()]
    print distances
    min_value = min(distances)
    print min_value

    return 0


def classify_row(row, means):
    return classify_markers(row['marker_1'], row['marker_2'], means)


def classify_df(df, means):
    return [classify_row(row, means) for index, row in df.iterrows()]


def classify(train, test):
    means = compute_means(train)
    classify_df(test, means)

    print means


def main():
    df = pd.read_csv('dataset_HW1.txt')
    children_data = df[df['patient_age'] < 18]

    children_train, children_test = train_test_split(children_data, test_size=0.3, random_state=109)
    classify(children_train, children_test)


if __name__ == '__main__':
    main()
