import math
from itertools import izip
from operator import itemgetter

import numpy as np
import pandas as pd


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


def split(df):
    test_size = 0.3
    length = df.shape[0]

    indices = list(df.index)
    np.random.shuffle(indices)

    test_length = int(round(test_size * length))
    test_indices = indices[:test_length]
    train_indices = indices[test_length:]

    test = df.loc[test_indices]
    train = df.loc[train_indices]

    return train, test


def evaluate_df(df):
    train, test = split(df)
    predicted_disease_subtypes = classify(train, test)
    actual_disease_subtypes = [row['subtype'] for index, row in test.iterrows()]
    correct_percentage = evaluate(actual_disease_subtypes, predicted_disease_subtypes)
    return correct_percentage


def evaluate_and_print(df, population_group_id):
    correct_percentage = evaluate_df(df)
    print 'percentage of the new patients who are correctly classified (%s): %.1f%%' % (
        population_group_id, 100.0 * correct_percentage)


def problem2():
    np.random.seed(109)

    df = pd.read_csv('dataset_HW1.txt')
    children_data = df[df['patient_age'] < 18]
    adult_women_data = df[(df['patient_age'] > 17) & (df['patient_gender'] == 'female')]
    adult_male_data = df[(df['patient_age'] > 17) & (df['patient_gender'] == 'male')]

    evaluate_and_print(children_data, 'children')
    evaluate_and_print(adult_women_data, 'adult women')
    evaluate_and_print(adult_male_data, 'adult men')


# ==== problem 3 ====================================================================================================


def classify_markers_prob3(marker1, marker2, train):
    # we iterate over the rows of the train dataframe and get the list of pairs: (subtype, distance)
    subtype_distance_pairs = [(row['subtype'], distance(marker1, marker2, row['marker_1'], row['marker_2']))
                              for index, row in train.iterrows()]
    # find the pair with the minimum distance
    min_pair = min(subtype_distance_pairs, key=itemgetter(1))
    # return its subtype
    return min_pair[0]


def classify_row_prob3(row, train):
    return classify_markers_prob3(row['marker_1'], row['marker_2'], train)


def classify_prob3(train, test):
    return [classify_row_prob3(row, train) for index, row in test.iterrows()]


def evaluate_df_prob3(df):
    train, test = split(df)
    predicted_disease_subtypes = classify_prob3(train, test)
    actual_disease_subtypes = [row['subtype'] for index, row in test.iterrows()]
    correct_percentage = evaluate(actual_disease_subtypes, predicted_disease_subtypes)
    return correct_percentage


def evaluate_and_print_prob3(df, population_group_id):
    correct_percentage = evaluate_df_prob3(df)
    print 'percentage of the new patients who are correctly classified (%s): %.1f%%' % (
        population_group_id, 100.0 * correct_percentage)


def problem3():
    # np.random.seed(1090)

    df = pd.read_csv('dataset_HW1.txt')
    children_data = df[df['patient_age'] < 18]
    adult_women_data = df[(df['patient_age'] > 17) & (df['patient_gender'] == 'female')]
    adult_male_data = df[(df['patient_age'] > 17) & (df['patient_gender'] == 'male')]

    evaluate_and_print_prob3(children_data, 'children')
    evaluate_and_print_prob3(adult_women_data, 'adult women')
    evaluate_and_print_prob3(adult_male_data, 'adult men')


def main():
    problem3()


if __name__ == '__main__':
    main()
