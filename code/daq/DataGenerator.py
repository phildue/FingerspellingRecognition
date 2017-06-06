import os
import random

import cv2
import numpy as np

from daq.preprocessing.PreProcessing import PreProcessor
from daq.preprocessing.SkinSegmentor import filter_skin


def gendata(dir_dataset, sample_size=2500, alphabet=None, sets=None):
    if sets is None:
        sets = ["A", "B", "C", "D", "E"]
    if alphabet is None:
        alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                    "v", "w", "x", "y"]

    n_letters = len(alphabet)
    im_res = (100, 120, 3)
    dim = im_res[0] * im_res[1]
    data = np.zeros(shape=(sample_size * n_letters, dim), dtype=np.uint8)
    labels = np.zeros(shape=(sample_size * n_letters, 1), dtype=np.uint8)
    pre_processor = PreProcessor(im_res)

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
            img = cv2.imread(path, 1)
            img = pre_processor.apply_pp(img)
            index = sel_samples + (class_ - 1) * sample_size
            data[index, :] = img.reshape(1, dim)
            labels[index] = class_

    return data, labels


def main():
    dir_dataset = '../../resource/dataset/fingerspelling5/dataset5/'
    n_data = 10
    data, labels = gendata(dir_dataset, n_data)
    print("Data:\n")
    print(data)
    print("Labels:\n")
    print(labels)

# main()
