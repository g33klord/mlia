import operator

import pandas as pd


def classify0(inx, dataset, labels, K=5):
    """K-Nearest Neighbors algorithm

    :param inx: input data .e.g [123, 34, 314]
    :param dataset: dataset (panda matrix) e.g.
            [ [123,1123,412], [123, 1665, 546], ......, [675,345,8676]
    :param labels: labeled vector corresponding to each row of dataset (panda matrix) e.g.
            [ "largeDoses", "smallDoses", "didntLike", ...... , "didntLike"]
    :param K: number if nearest neighbor (+ve integer)
    :return:
    """

    dataset_size = dataset.shape[0]  # no of rows

    # make a matrix of same size of dataset to find difference of input and each and every dataset
    inx_matrix = pd.np.tile(inx, (dataset_size, 1))  # tile(vector, (row_len, column_len))

    diff_matrix = inx_matrix - dataset
    square_diff_matrix = diff_matrix ** 2
    square_distance = square_diff_matrix.sum(axis=1)
    distance = square_distance ** 0.5

    sorted_distance_indicies = distance.argsort()

    class_count = {}
    for i in range(K):
        vote_label = labels.iloc[sorted_distance_indicies.iloc[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    # short dictionary `class_count` in decreasing order by its value (high vote to low vote)
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    winner_class = sorted_class_count[0][0]
    return winner_class


def file2matrix(filename):
    """
    parse dataset text file and return dataset and label vector
    :param filename: local filepath or web address to file
    :return: dataset, labels
    """
    header_names = ["frequentFlyerMiles", "VideoGamePlayedHour", "IceCreamEatenLiter", "labels"]
    dataset = pd.read_csv(filename, sep='\t', names=header_names)  # provided sample is tab separated

    features_matrix = dataset.loc[:, : "IceCreamEatenLiter"]
    label_vector = dataset.loc[:, "labels"]

    return features_matrix, label_vector


def auto_norm(dataset):
    """
    Automatically normalize dataset to value between 0 and 1
    :param dataset: panda dataframe or numpy array
    :return: (normalized_dataset, ranges, min_values)
    """
    min_values = dataset.min()
    max_values = dataset.max()
    ranges = max_values - min_values
    m = dataset.shape[0]
    normalized_dataset = dataset - pd.np.tile(min_values, (m, 1))
    normalized_dataset = normalized_dataset / pd.np.tile(ranges, (m, 1))
    return normalized_dataset, ranges, min_values


def dating_class_test():
    ho_ratio = 0.10
    dating_data_matrix, dating_labels = file2matrix('datingTestSet')
    normalized_matrix, ranges, min_values = auto_norm(dating_data_matrix)
    m = normalized_matrix.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0

    for i in range(num_test_vecs):
        classifier_result = classify0(
            normalized_matrix.loc[i, :],
            normalized_matrix.loc[num_test_vecs:m, :],
            dating_labels.loc[num_test_vecs:m], 3
        )
        print(f"Result: {classifier_result}, Expected: {dating_labels[i]}")
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print(f"Total error rate is {error_count/float(num_test_vecs)}")

