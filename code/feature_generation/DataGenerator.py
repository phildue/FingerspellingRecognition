import os
import random

import cv2
import numpy as np


def gendata(dir_dataset, sample_size):
    sets = ["A", "B", "C", "D", "E"]
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
                "w", "x", "y"]
    n_letters = len(alphabet)
    im_res = (100, 120, 3)
    dim = im_res[0] * im_res[1] * im_res[2]
    data = np.zeros(shape=(sample_size * n_letters, dim))
    labels = np.zeros(shape=(sample_size * n_letters, 1))

    for class_, dir_letter in enumerate(alphabet, 1):
        paths = []
        for dir_set in sets:
            fnames = [f for f in os.listdir(dir_dataset + dir_set + "/" + dir_letter) if 'color' in f]
            for fname in fnames:
                paths.append(dir_dataset + dir_set + "/" + dir_letter + "/" + fname)

        for sel_samples, path in enumerate(random.sample(paths, sample_size)):
            img = cv2.imread(path, 1)
            img = cv2.resize(img, (im_res[0], im_res[1]))
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


#main()
