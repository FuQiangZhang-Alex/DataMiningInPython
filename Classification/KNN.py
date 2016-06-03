# k-Nearest Neighbors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

labels = {
    '1': 'Working at Computer',
    '2': 'Standing Up, Walking and Going up/down stairs',
    '3': 'Standing',
    '4': 'Walking',
    '5': 'Going Up/Down Stairs',
    '6': 'Walking and Talking with Someone',
    '7': 'Talking while Standing'
}


def knn_classify(in_x, data_set, lbl, k):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set  # difference of input data and all training data
    square_diff_mat = diff_mat ** 2
    distance_mat = square_diff_mat ** 0.5
    sorted_distance = distance_mat.argsort()
    print(sorted_distance)


def file2matrix():
    class_vector = []
    with open(file='data/Activity Recognition from Single Chest-Mounted Accelerometer/debug.csv') as training:
        index = 0
        line_total = len(training.readlines())
        mat = np.zeros((line_total, 3))
    with open(file='data/Activity Recognition from Single Chest-Mounted Accelerometer/debug.csv') as training:
        for index, line in enumerate(training.readlines()):
            record = line.split(',')
            mat[index, :] = record[1:4]
            class_vector.append(int(record[4].rstrip('\n')))
    return mat, class_vector


def draw():
    training_set, class_array = file2matrix()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=training_set[0:100, 0], ys=training_set[0:100, 1], zs=training_set[0:100, 2]
               , c=np.array(class_array) / 10, s=5)
    plt.show()


def norm_data(data_set):
    data_set = np.array(data_set)  # nd array
    min_val = data_set.min(0)
    max_val = data_set.max(0)
    ranges = max_val - min_val  # range
    norm_data_set = np.zeros(np.shape(data_set))  # initialize it with zeros
    m = data_set.shape[0]  # m rows
    norm_data_set = data_set - np.tile(min_val, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, min_val, max_val

matrix, class_array = file2matrix()
normal_matrix, min_v, max_v = norm_data(matrix)
inx = [1866, 2390, 2282]
knn_classify(inx, matrix, None, None)
# print(matrix)
# print(min_v, max_v)
# print(normal_matrix)
