import os
import numpy as np


def get_root_project_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


def get_word_list():
    word_list = []
    with open("./data/words_list.txt", "r") as input:
        for line in input.readlines():
            word_list.append(line.replace("\n", ""))
    return word_list


def compute_distance(point_a, point_b):
    return np.sqrt(np.sum((point_a - point_b) ** 2, axis=0))


def get_mean(*args):
    return np.array([sum([elem[0] for elem in args])/len(args), sum([elem[1] for elem in args])/len(args),
           sum([elem[2] for elem in args])/len(args)])
