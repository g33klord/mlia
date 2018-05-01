import os
import operator

import pandas as pd


def classify(inx, dataset, labels, K=5):
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
    square_distance = square_diff_matrix.sum(axis=1)  # axis 1 = column
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
    dataset = pd.read_csv(filename, sep='\t', header=None)  # provided sample is tab separated

    features_matrix = dataset.loc[:, : 2]
    label_vector = dataset.loc[:, 3]
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
    dating_data_matrix, dating_labels = file2matrix('datingTestSet2')
    normalized_matrix, ranges, min_values = auto_norm(dating_data_matrix)
    m = normalized_matrix.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0

    for i in range(num_test_vecs):
        classifier_result = classify(
            normalized_matrix.iloc[i, :],
            normalized_matrix.iloc[num_test_vecs:m, :],
            dating_labels.iloc[num_test_vecs:m], 3
        )
        print(f"Result: {classifier_result}, Expected: {dating_labels[i]}")
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print(f"Total error rate is {error_count/float(num_test_vecs)}")


def classify_person():
    game_percentage = float(input("Percentage of time spending playing video game?"))
    ffmiles = float(input("Frequent flyer mile earned per year?"))
    icecream = float(input("Liters of ice cream consumed per year?"))
    dating_marix, dating_label = file2matrix('datingTestSet')
    normalized_matrix, ranged, min_values = auto_norm(dating_marix)

    input_array = pd.np.array([ffmiles, game_percentage, icecream])

    classifier_result = classify(input_array, normalized_matrix, dating_label)
    print(f"Result: {classifier_result}")


def image2vector(filename):
    """
    convert image text file to vector
    :param filename:
    :return: 1X1024 vector
    """
    dataframe = pd.read_fwf(filename, widths=[1] * 32, header=None)  # 32 X 32 matrix
    # http://pandas.pydata.org/pandas-docs/version/0.20/generated/pandas.read_fwf.html
    return pd.np.ravel(dataframe)  # 1 X 1024 matrix


def handwriting_class_test():
    # preparing Training dataset
    print('preparing Training dataset')
    tranining_file_list = os.listdir("digits/trainingDigits")
    m = len(tranining_file_list)
    hw_labels = pd.Series(index=range(m))
    training_matrix = pd.DataFrame(index=range(m), columns=range(1024))

    for i in range(m):
        file_name = tranining_file_list[i]
        print('.', end='')
        print('')
        # extract digit class from file name (first char of filename is the digit)
        digit_class = int(file_name[0])

        hw_labels.loc[i, 0] = digit_class
        training_matrix.loc[i, :] = image2vector(f'digits/trainingDigits/{file_name}')

    # Testing
    print('Testing')
    test_file_list = os.listdir("digits/testDigits")
    error_count = 0.0
    m_test = len(test_file_list)

    for i in range(m_test):
        file_name = test_file_list[i]
        digit_class = int(file_name[0])
        vector_under_test = image2vector(f'digits/testDigits/{file_name}')
        classifier_result = classify(vector_under_test, training_matrix, hw_labels, 3)

        print(f'Result: {classifier_result}, Expected: {digit_class}')

        if classifier_result != digit_class:
            error_count += 1.0
    print(f'Total Error: {int(error_count)}')
    print(f'Error rate: {error_count/float(m_test)}')
