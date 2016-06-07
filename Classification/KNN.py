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
    square_distances = square_diff_mat.sum(axis=1)
    distance_mat = square_distances ** 0.5
    sorted_distance = distance_mat.argsort()
    class_count = {}
    for i in range(k):
        vote_label = lbl[sorted_distance[i]]
        class_count[str(vote_label)] = labels.get(str(vote_label), 'None')
    return class_count


def file2matrix(file_name):
    class_vector = []
    with open(file=file_name) as training:
        index = 0
        line_total = len(training.readlines())
        mat = np.zeros((line_total, 3))
    with open(file=file_name) as training:
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

matrix, class_array = file2matrix('data/Activity Recognition from Single Chest-Mounted Accelerometer/debug.csv')
# normal_matrix, min_v, max_v = norm_data(matrix)
# for data, class_name in zip(matrix, class_array):
#    print(data, class_name, labels.get(str(class_name)))
inx = np.array([2089, 2539, 2158])
class_cnt = knn_classify(inx, matrix, class_array, 10)
print(class_cnt)
