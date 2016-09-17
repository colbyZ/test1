import bisect

import numpy as np
import pandas as pd


def split(data, m):
    test_size = 1 - m
    length = len(data)

    indices = list(data.index)
    np.random.shuffle(indices)

    test_length = int(round(test_size * length))
    test_indices = indices[:test_length]
    train_indices = indices[test_length:]

    test = data.loc[test_indices]
    train = data.loc[train_indices]

    return train, test


def distance(x1, x2):
    return abs(x1 - x2)


def find_best_neighbors(sorted_x_list, neighbors_range, test_x):
    current_left_index = neighbors_range[0]
    current_right_index = neighbors_range[1]
    best_range = current_left_index, current_right_index
    while current_right_index < len(sorted_x_list):
        left_distance = distance(test_x, sorted_x_list[current_left_index])
        right_distance = distance(test_x, sorted_x_list[current_right_index])
        if left_distance <= right_distance:
            break
        best_range = current_left_index, current_right_index
        current_left_index += 1
        current_right_index += 1
    if best_range[0] == neighbors_range[0]:
        # we didn't find better neighbors on the right, so we'll search left
        current_left_index = neighbors_range[0] - 1
        current_right_index = neighbors_range[1] - 1
        while current_left_index >= 0:
            left_distance = distance(test_x, sorted_x_list[current_left_index])
            right_distance = distance(test_x, sorted_x_list[current_right_index])
            if left_distance >= right_distance:
                break
            best_range = current_left_index, current_right_index
            current_left_index -= 1
            current_right_index -= 1
    return best_range


def find_nearest_neighbors(k, sorted_x_list, test_x):
    insertion_index = bisect.bisect_left(sorted_x_list, test_x)
    initial_left_index = max(0, insertion_index - k / 2)
    neighbors_range = (initial_left_index, initial_left_index + k)
    best_neighbors = find_best_neighbors(sorted_x_list, neighbors_range, test_x)
    print neighbors_range, best_neighbors


def knn_predict_one_point(k, sorted_train, sorted_x_list, test_x):
    find_nearest_neighbors(k, sorted_x_list, test_x)
    return 0.0


def knn_predict(k, train, test):
    sorted_train = train.sort_values(by='x')
    sorted_x_list = sorted_train['x'].tolist()
    predicted_test = test.copy()

    for i, (index, row) in enumerate(test.iterrows()):
        knn_predict_one_point(k, sorted_train, sorted_x_list, row['x'])
        if i == 0:
            break

    # predicted_test['y'] = [knn_predict_one_point(k, sorted_train, sorted_x_list, row['x'])
    #                        for index, row in test.iterrows()]
    return predicted_test


def compare_with_sklearn():
    np.random.seed(1090)

    df = pd.read_csv('dataset/dataset_1_full.txt')

    train, test = split(df, 0.7)

    predicted_test = knn_predict(1, train, test[['x']])
    # print predicted_test


if __name__ == '__main__':
    compare_with_sklearn()
