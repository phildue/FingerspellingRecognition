import os
import random

import numpy as np

from daq.datageneration.ImReader import read_im_file
from daq.preprocessing.PreProcessing import pre_processing


def gendata_sign(dir_dataset='../../resource/dataset/fingerspelling5/dataset5/',
                 sample_size=2500, alphabet=None, sets=None, im_size=(100, 120, 3)):
    if sets is None:
        sets = ["A", "B", "C", "D", "E"]
    if alphabet is None:
        alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                    "v", "w", "x", "y"]

    n_letters = len(alphabet)
    dim = im_size[0] * im_size[1] * im_size[2]
    data = np.zeros(shape=(sample_size * n_letters, dim), dtype=np.uint8)
    labels = np.zeros(shape=(sample_size * n_letters, 1), dtype=np.uint8)

    for class_, dir_letter in enumerate(alphabet, 1):
        paths = []
        for dir_set in sets:
            fnames = [f for f in os.listdir(dir_dataset + dir_set + "/" + dir_letter) if 'color' in f]
            for fname in fnames:
                paths.append(dir_dataset + dir_set + "/" + dir_letter + "/" + fname)

        path_sel = paths
        try:
            path_sel = random.sample(paths, sample_size)
        except ValueError:
            print("Too many samples requested for " + dir_letter + ", taking all: " + str(len(path_sel)))

        for sel_samples, path in enumerate(path_sel):
            img = read_im_file(path)
            img = pre_processing(img, im_size)
            index = sel_samples + (class_ - 1) * sample_size
            data[index, :] = img.reshape(1, dim)
            labels[index] = class_

    return data[1:sample_size, :], labels[1:sample_size, :]


def demo():
    dir_dataset = '../../resource/dataset/fingerspelling5/dataset5/'
    n_data = 10
    data, labels = gendata_sign(dir_dataset, n_data)
    print("Data:\n")
    print(data)
    print("Labels:\n")
    print(labels)

# demo()
