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


def find_best_neighbors_range(sorted_x_list, neighbors_range, test_x):
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


def get_initial_range(k, insertion_index, length):
    left_index = max(0, insertion_index - k / 2)
    right_index = left_index + k
    if right_index > length:
        right_index = length
        left_index = right_index - k
    return left_index, right_index


def find_nearest_neighbors(k, sorted_x_list, test_x):
    insertion_index = bisect.bisect_left(sorted_x_list, test_x)
    initial_range = get_initial_range(k, insertion_index, len(sorted_x_list))
    return find_best_neighbors_range(sorted_x_list, initial_range, test_x)


def knn_predict_one_point(k, sorted_train, sorted_x_list, test_x):
    neighbors_range = find_nearest_neighbors(k, sorted_x_list, test_x)
    total = 0.0
    for index in neighbors_range:
        total += sorted_train.iloc[index]['y']
    return total / k


def knn_predict(k, train, test):
    sorted_train = train.sort_values(by='x')
    sorted_x_list = sorted_train['x'].tolist()
    predicted_test = test.copy()

    predicted_test['y'] = [knn_predict_one_point(k, sorted_train, sorted_x_list, row['x'])
                           for index, row in test.iterrows()]
    return predicted_test


def compare_with_sklearn():
    np.random.seed(1090)

    df = pd.read_csv('dataset/dataset_1_full.txt')

    train, test = split(df, 0.7)

    predicted_test = knn_predict(1, train, test[['x']])
    print predicted_test


if __name__ == '__main__':
    compare_with_sklearn()
