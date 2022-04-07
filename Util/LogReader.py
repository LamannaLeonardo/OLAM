# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import Configuration


def preprocess(data):

        filtered = [str.replace("\t\t", " ").replace("|","").split() for str in data
                    if str.strip() != "" and all([el.isnumeric()
                    for el in str.replace("\t\t", " ").replace(".","").replace("|","").split()])]

        filtered = np.array(filtered)

        times = [float(str) for str in filtered[:, 0]]
        iterations = [int(str) for str in filtered[:, 1]]
        real_precs_size = [int(str) for str in filtered[:, 2]]
        learned_precs_size = [int(str) for str in filtered[:, 3]]
        real_eff_pos_size = [int(str) for str in filtered[:, 4]]
        learned_eff_pos_size = [int(str) for str in filtered[:, 5]]
        real_eff_neg_size = [int(str) for str in filtered[:, 6]]
        learned_eff_neg_size = [int(str) for str in filtered[:, 7]]
        ins_pre = [int(str) for str in filtered[:, 8]]
        del_pre = [int(str) for str in filtered[:, 9]]
        ins_eff_pos = [int(str) for str in filtered[:, 10]]
        del_eff_pos = [int(str) for str in filtered[:, 11]]
        ins_eff_neg = [int(str) for str in filtered[:, 12]]
        del_eff_neg = [int(str) for str in filtered[:, 13]]
        precs_recall = [float(str) for str in filtered[:, 14]]
        eff_pos_recall = [float(str) for str in filtered[:, 15]]
        eff_neg_recall = [float(str) for str in filtered[:, 16]]
        precs_precision = [float(str) for str in filtered[:, 17]]
        eff_pos_precision = [float(str) for str in filtered[:, 18]]
        eff_neg_precision = [float(str) for str in filtered[:, 19]]
        overall_recall = [float(str) for str in filtered[:, 20]]
        overall_precision = [float(str) for str in filtered[:, 21]]

        return times, iterations, real_precs_size, learned_precs_size, real_eff_pos_size, learned_eff_pos_size, \
           real_eff_neg_size, learned_eff_neg_size, ins_pre, del_pre, ins_eff_pos, del_eff_pos, \
           ins_eff_neg, del_eff_neg, precs_recall, eff_pos_recall, eff_neg_recall, precs_precision, \
           eff_pos_precision, eff_neg_precision, overall_recall, overall_precision


def preprocess_FF_time(data):
    filtered = []

    for i in range(len(data)):
        str = data[i]

        if str.find("FF computational") == 0 or str.find("FD computational") == 0:
            mystr = ":".join(str.split(":")[1:]).strip()
            d = datetime.datetime.strptime(mystr, "%H:%M:%S.%f")
            filtered.append((d - datetime.datetime(1900, 1, 1)).total_seconds())

    return filtered


def preprocess_iter_time(data):
    filtered = []

    for i in range(len(data)):
        str = data[i]

        if str.find("Iteration computational") == 0:
            mystr = ":".join(str.split(":")[1:]).strip()
            d = datetime.datetime.strptime(mystr, "%H:%M:%S.%f")
            filtered.append((d - datetime.datetime(1900, 1, 1)).total_seconds())

    return filtered


def plot_FF_time(log_file):

    instance_name = Configuration.INSTANCE_DATA_PATH_PDDL.split("/")[-1].split(".")[0]

    with open(log_file, 'r') as file:
        data = file.read().split('\n')

        FF_time = preprocess_FF_time(data)

        if len(FF_time) == 0:
            return

        dataframe = pd.DataFrame({
            "FD iterations": range(1, len(FF_time) + 1),
            "FD time": FF_time
        })

        ax = dataframe.plot(x="FD iterations", y="FD time", title="Planning time of instance {}".format(instance_name))

        ax.set_xlabel("FD iterations")
        ax.set_ylabel("Time (seconds)")
        ax.legend()

        plt.legend()

        plt.savefig(os.path.join(log_file[:log_file.index(log_file.split("/")[-1])], instance_name + "_planningTime.png"))

        plt.close()


def plot_iter_time(log_file):

    instance_name = Configuration.INSTANCE_DATA_PATH_PDDL.split("/")[-1].split(".")[0]

    with open(log_file, 'r') as file:
        data = file.read().split('\n')

        iter_time = preprocess_iter_time(data)

        dataframe = pd.DataFrame({
            "Iterations": range(1, len(iter_time) + 1),
            "Iteration time": iter_time
        })

        ax = dataframe.plot(x="Iterations", y="Iteration time", title="Iteration time of instance {}".format(instance_name))

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Time (seconds)")
        ax.legend()

        plt.legend()


        plt.savefig(os.path.join(log_file[:log_file.index(log_file.split("/")[-1])], instance_name + "_iterationTime.png"))

        plt.close()


def plot_recall(log_file):

    instance_name = Configuration.INSTANCE_DATA_PATH_PDDL.split("/")[-1].split(".")[0]

    with open(log_file, 'r') as file:
        data = file.read().split('\n')

        times, iterations, real_precs_size, learned_precs_size, real_eff_pos_size, learned_eff_pos_size, \
        real_eff_neg_size, learned_eff_neg_size, ins_pre, del_pre, ins_eff_pos, del_eff_pos, \
        ins_eff_neg, del_eff_neg, precs_recall, eff_pos_recall, eff_neg_recall, precs_precision, \
        eff_pos_precision, eff_neg_precision, overall_recall, overall_precision = preprocess(data)

        dataframe = pd.DataFrame({
            "Time": times,
            "Iterations": iterations,
            "Overall recall": overall_recall
        })

        ax = dataframe.plot(x="Iterations", y="Overall recall", title="Recall of instance {}".format(instance_name))

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Overall recall")

        plt.savefig(os.path.join(log_file[:log_file.index(log_file.split("/")[-1])], instance_name + "_overall_recall.png"))

        plt.close()
        # plt.show()


def plot_precision(log_file):

    instance_name = Configuration.INSTANCE_DATA_PATH_PDDL.split("/")[-1].split(".")[0]

    with open(log_file, 'r') as file:
        data = file.read().split('\n')

        times, iterations, real_precs_size, learned_precs_size, real_eff_pos_size, learned_eff_pos_size, \
        real_eff_neg_size, learned_eff_neg_size, ins_pre, del_pre, ins_eff_pos, del_eff_pos, \
        ins_eff_neg, del_eff_neg, precs_recall, eff_pos_recall, eff_neg_recall, precs_precision, \
        eff_pos_precision, eff_neg_precision, overall_recall, overall_precision = preprocess(data)

        dataframe = pd.DataFrame({
            "Time": times,
            "Iterations": iterations,
            "Overall precision": overall_precision
        })

        ax = dataframe.plot(x="Iterations", y="Overall precision", title="Precision of instance {}".format(instance_name))

        ax.set_xlabel("Iterations")
        ax.set_ylabel("Overall precision")

        plt.savefig(os.path.join(log_file[:log_file.index(log_file.split("/")[-1])], instance_name + "_overall_precision.png"))

        plt.close()
        # plt.show()


def evaluate_log_metrics(log_file):

    # Plot overall precision
    # plot_precision(log_file)

    # Plot overall recall
    # plot_recall(log_file)

    # Plot planning time
    plot_FF_time(log_file)
