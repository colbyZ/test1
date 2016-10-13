from itertools import izip

import numpy as np


def score(predicted, actual):
    rss = 0.0
    tss = 0.0
    actual_y_list = actual['y']
    actual_y_mean = np.mean(actual_y_list)
    for predicted_value, actual_value in izip(predicted['y'], actual_y_list):
        rss += (actual_value - predicted_value) ** 2
        tss += (actual_value - actual_y_mean) ** 2
    return 1.0 - rss / tss


def get_predictions(xs, intercept, slope):
    for x in xs:
        yield intercept + slope * x


def calc_r_sq(xs, ys, intercept, slope):
    predictions = list(get_predictions(xs, intercept, slope))
    rss = 0.0
    tss = 0.0
    actual_y_mean = np.mean(ys)
    for predicted_value, actual_value in izip(predictions, ys):
        rss += (actual_value - predicted_value) ** 2
        tss += (actual_value - actual_y_mean) ** 2
    return 1.0 - rss / tss


def q_7():
    n = 100

    r_sq_list = []
    for _ in xrange(500):
        xs = np.random.uniform(0.0, 10.0, n)
        ys = 10.0 + 2.0 * xs + np.random.uniform(-5.0, 5.0, n)

        r_sq = calc_r_sq(xs, ys, 15.0, 1.0)
        r_sq_list.append(r_sq)

    print '30 - 2x, R^2: %.4f' % np.mean(r_sq_list)


def main():
    q_7()


if __name__ == '__main__':
    main()
