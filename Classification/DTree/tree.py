
# using ID3 algorithm to calculate
from math import log
from Classification import *


def calculate_shannon_entropy(data_set):
    """
    measure the information before and after the split
    The change in information before and after the split is known as the information gain.
    This function calculate the information gain
    The split with the highest information gain is your best option.
    Shannon entropy - The measure of information of a set
    :param data_set: calculate information gain against
    :return:
    """
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        print(feat_vec)
        current_label = feat_vec[-1]
        if current_label not in label_counts:
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


def split_data_set(data_set, axis, value):
    """
    Data set splitting on a given feature
    :param data_set: data set we’ll split
    :param axis: the feature will split on
    :param value: the value of the feature to return
    :return: split data set
    """
    sub_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = list(feat_vec[:axis])
            reduced_feat_vec.extend(feat_vec[axis+1:])
            sub_set.append(reduced_feat_vec)
    return sub_set


data = file2list(file='data/car.csv')
print(calculate_shannon_entropy(data))
