import math
from itertools import izip
from operator import itemgetter

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
    index_distance_pairs = [(index, distance(marker1, marker2, row['marker_1'], row['marker_2']))
                            for index, row in means.iterrows()]
    min_pair = min(index_distance_pairs, key=itemgetter(1))
    return min_pair[0]


def classify_row(row, means):
    return classify_markers(row['marker_1'], row['marker_2'], means)


def classify_df(df, means):
    return [classify_row(row, means) for index, row in df.iterrows()]


def classify(train, test):
    means = compute_means(train)
    predicted_disease_subtypes = classify_df(test, means)
    return predicted_disease_subtypes


def evaluate(actual_disease_subtypes, predicted_disease_subtypes):
    correct_count = 0
    for actual_subtype, predicted_subtype in izip(actual_disease_subtypes, predicted_disease_subtypes):
        correct_count += (1 if actual_subtype == predicted_subtype else 0)
    correct_percentage = 1.0 * correct_count / len(actual_disease_subtypes)
    return correct_percentage


def main():
    df = pd.read_csv('dataset_HW1.txt')
    children_data = df[df['patient_age'] < 18]

    children_train, children_test = train_test_split(children_data, test_size=0.3, random_state=109)
    predicted_disease_subtypes = classify(children_train, children_test)
    actual_disease_subtypes = [row['subtype'] for index, row in children_test.iterrows()]

    correct_percentage = evaluate(actual_disease_subtypes, predicted_disease_subtypes)
    print 'percentage of the new patients who are correctly classified: %.2f%%' % (100.0 * correct_percentage)


if __name__ == '__main__':
    main()
