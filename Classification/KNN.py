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

draw()
