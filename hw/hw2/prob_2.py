import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression as Lin_Reg
from sklearn.neighbors import KNeighborsRegressor as KNN


def fill(regressor, dataset_data):
    missing_df, full_df, no_y_ind, with_y_ind = dataset_data
    # preparing data in array form
    x_train = missing_df.loc[with_y_ind, 'x'].values
    x_train = x_train.reshape((len(with_y_ind), 1))
    y_train = missing_df.loc[with_y_ind, 'y'].values
    x_test = missing_df.loc[no_y_ind, 'x'].values.reshape((len(no_y_ind), 1))
    y_test = full_df.loc[no_y_ind, 'y'].values

    # fit model
    regressor.fit(x_train, y_train)

    # predict y-values
    predicted_y = regressor.predict(x_test)

    # score predictions
    r = regressor.score(x_test, y_test)

    # fill in missing y-values
    predicted_df = missing_df.copy()
    predicted_df.loc[no_y_ind, 'y'] = pd.Series(predicted_y, index=no_y_ind)

    return predicted_df, r


def scatter(ax, predicted_df, indices, color):
    ax.scatter(predicted_df.loc[indices]['x'].values,
               predicted_df.loc[indices]['y'].values,
               color=color)


def plot_ax(ax, predicted_df, dataset_data, no_ind_color, title):
    no_y_ind = dataset_data[2]
    with_y_ind = dataset_data[3]
    scatter(ax, predicted_df, with_y_ind, 'blue')
    scatter(ax, predicted_df, no_y_ind, no_ind_color)
    ax.set_title(title)


def plot_missing(ax1, ax2, predicted_knn, r_knn, predicted_lin, r_lin, dataset_data, dataset_i):
    plot_ax(ax1, predicted_knn, dataset_data, 'red', 'Dataset %d, KNN, R^2: %.3f' % (dataset_i, r_knn))
    plot_ax(ax2, predicted_lin, dataset_data, 'green', 'Lin Reg, R^2: %.3f' % r_lin)


def get_dataset_data(dataset_i):
    # Read dataset i
    missing_df = pd.read_csv('./dataset/dataset_%d_missing.txt' % dataset_i)
    full_df = pd.read_csv('./dataset/dataset_%d_full.txt' % dataset_i)

    no_y_ind = missing_df[missing_df['y'].isnull()].index
    with_y_ind = missing_df[missing_df['y'].notnull()].index
    return missing_df, full_df, no_y_ind, with_y_ind


def handling_missing_data():
    # number of neighbours
    k = 10

    n_datasets = 6

    # plot predicted points
    fig, ax_pairs = plt.subplots(n_datasets, 2, figsize=(15, 3.3 * n_datasets))

    for dataset_i in range(1, n_datasets + 1):
        dataset_data = get_dataset_data(dataset_i)

        predicted_knn, r_knn = fill(KNN(n_neighbors=k), dataset_data)
        predicted_lin, r_lin = fill(Lin_Reg(), dataset_data)

        ax_pair = ax_pairs[dataset_i - 1]
        plot_missing(ax_pair[0], ax_pair[1],
                     predicted_knn, r_knn, predicted_lin,
                     r_lin, dataset_data, dataset_i)

    plt.tight_layout()
    plt.show()


def impact_of_k_on_knn():
    dataset_data = get_dataset_data(1)
    k_list = []
    r_sq_list = []
    for k in range(1, 346):
        _, r_knn = fill(KNN(n_neighbors=k), dataset_data)
        k_list.append(k)
        r_sq_list.append(r_knn)
    plt.plot(k_list, r_sq_list)
    plt.xlabel('k')
    plt.ylabel('r squared')
    plt.title('impact of k on the performance of KNN, k: [1, 345]')
    plt.show()


if __name__ == '__main__':
    # handling_missing_data()
    impact_of_k_on_knn()
