import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression as Lin_Reg
from sklearn.neighbors import KNeighborsRegressor as KNN


# --------  fill_knn
# input: missing_df (dataframe), full_df (dataframe), no_y_ind (indices of missing values),
#       with_y_ind (indices of non-missing values), k (integer)
# output: predicted_df (dataframe), r (float)

def fill_knn(missing_df, full_df, no_y_ind, with_y_ind, k):
    # preparing data in array form
    x_train = missing_df.loc[with_y_ind, 'x'].values
    x_train = x_train.reshape((len(with_y_ind), 1))
    y_train = missing_df.loc[with_y_ind, 'y'].values
    x_test = missing_df.loc[no_y_ind, 'x'].values.reshape((len(no_y_ind), 1))
    y_test = full_df.loc[no_y_ind, 'y'].values

    # fit knn model
    neighbours = KNN(n_neighbors=k)
    neighbours.fit(x_train, y_train)

    # predict y-values
    predicted_y = neighbours.predict(x_test)

    # score predictions
    r = neighbours.score(x_test, y_test)

    # fill in missing y-values
    predicted_df = missing_df.copy()
    predicted_df.loc[no_y_ind, 'y'] = pd.Series(predicted_y, index=no_y_ind)

    return predicted_df, r


# --------  fill_ling_reg
# input: missing_df (dataframe), full_df (dataframe), no_y_ind (indices of missing values),
#       with_y_ind (indices of non-missing values), k (integer)
# output: predicted_df (dataframe), r (float)

def fill_lin_reg(missing_df, full_df, no_y_ind, with_y_ind):
    # preparing data in array form
    x_train = missing_df.loc[with_y_ind, 'x'].values.reshape((len(with_y_ind), 1))
    y_train = missing_df.loc[with_y_ind, 'y'].values
    x_test = missing_df.loc[no_y_ind, 'x'].values.reshape((len(no_y_ind), 1))
    y_test = full_df.loc[no_y_ind, 'y'].values

    # fit linear model
    regression = Lin_Reg()
    regression.fit(x_train, y_train)

    # predict y-values
    predicted_y = regression.predict(x_test)

    # score predictions
    r = regression.score(x_test, y_test)

    # fill in missing y-values
    predicted_df = missing_df.copy()
    predicted_df.loc[no_y_ind, 'y'] = pd.Series(predicted_y, index=no_y_ind)

    return predicted_df, r


# --------  plot_missing
# input: ax1 (axes), ax2 (axes),
#       predicted_knn (nx2 dataframe with predicted vals), r_knn (float),
#       predicted_lin (nx2 dataframe with predicted vals), r_lin (float),
#       k (integer),
#       no_y_ind (indices of rows with missing y-values),
#       with_y_ind (indices of rows with no missing y-values)

def plot_missing(ax1, ax2, predicted_knn, r_knn, predicted_lin, r_lin, k, no_y_ind, with_y_ind, dataset_i):
    ax1.scatter(predicted_knn.loc[with_y_ind]['x'].values,
                predicted_knn.loc[with_y_ind]['y'].values,
                color='blue')

    ax1.scatter(predicted_knn.loc[no_y_ind]['x'].values,
                predicted_knn.loc[no_y_ind]['y'].values,
                color='red')

    ax1.set_title('Dataset %d, KNN, R^2: %.3f' % (dataset_i, r_knn))

    ax2.scatter(predicted_lin.loc[with_y_ind]['x'].values,
                predicted_lin.loc[with_y_ind]['y'].values,
                color='blue')

    ax2.scatter(predicted_lin.loc[no_y_ind]['x'].values,
                predicted_lin.loc[no_y_ind]['y'].values,
                color='green')

    ax2.set_title('Lin Reg, R^2: %.3f' % r_lin)


def handling_missing_data():
    # number of neighbours
    k = 10

    n_datasets = 6

    # plot predicted points
    fig, ax_pairs = plt.subplots(n_datasets, 2, figsize=(15, 3.3 * n_datasets))

    for dataset_i in range(0, n_datasets):
        # Read dataset i
        dataset_i_1 = dataset_i + 1
        missing_df = pd.read_csv('./dataset/dataset_%d_missing.txt' % dataset_i_1)
        full_df = pd.read_csv('./dataset/dataset_%d_full.txt' % dataset_i_1)

        no_y_ind = missing_df[missing_df['y'].isnull()].index
        with_y_ind = missing_df[missing_df['y'].notnull()].index

        predicted_knn, r_knn = fill_knn(missing_df, full_df, no_y_ind, with_y_ind, k)
        predicted_lin, r_lin = fill_lin_reg(missing_df, full_df, no_y_ind, with_y_ind)

        ax_pair = ax_pairs[dataset_i]
        plot_missing(ax_pair[0], ax_pair[1],
                     predicted_knn, r_knn,
                     predicted_lin, r_lin,
                     k, no_y_ind, with_y_ind, dataset_i_1)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    handling_missing_data()
