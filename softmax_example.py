import numpy as np
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

MODELS_DIR = "models/"
LOGS_DIR = "logs/"
PLOTS_DIR = "plots/"


def my_softmax(data_vector):
    return [(np.exp(j) / np.sum(np.exp(i) for i in data_vector)) for j in data_vector]


if __name__ == "__main__":
    # data_vector = [0.23, 0.34, 0.17, 0.82]
    data_vector = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                   0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8]
    # data_vector2 = [i*10 for i in data_vector]

    softmax_vector = my_softmax(data_vector)
    print(softmax_vector)
    print('The softmax sum is equal to: {}'.format(np.sum(softmax_vector)))


