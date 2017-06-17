import os
import random

import numpy as np

from daq.ImReader import read_image, read_letters
from daq.preprocessing.PreProcessing import pre_processing


def gendata_skin(path_dataset='../../resource/dataset/skin/Skin_NonSkin.txt',
                 sample_size=245000):
    data = np.zeros(shape=(sample_size, 3))
    labels = np.zeros(shape=(sample_size, 1))

    with open(path_dataset) as f:
        for n, line in enumerate(f):
            if n >= sample_size:
                break
            elements = line.split()
            data[n, :] = np.array(list(map(int, elements[0:3])))
            labels[n, 0] = np.array(list(map(int, elements[3])))

    return data[1:sample_size, :], labels[1:sample_size, :]


def gendata_sign(img_file_paths,
                 sample_size=2500, letter_imgs=None):
    if letter_imgs is None:
        letter_imgs = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                    "v", "w", "x", "y"]
    letter_imgs = read_letters(img_file_paths, sample_size, letter_imgs)

    for letter in letter_imgs:
        for i, img in enumerate(letter_imgs[letter]):
            letter_imgs[letter][i] = pre_processing(img)

    n_letters = len(letter_imgs)
    dim = list(letter_imgs.values())[0][0][0] * list(letter_imgs.values())[0][0][1]
    data = np.zeros(shape=(sample_size * n_letters, dim), dtype=np.uint8)
    labels = np.zeros(shape=(sample_size * n_letters, 1), dtype=np.uint8)
    for class_, letter in enumerate(letter_imgs.keys(), 1):
        for n, image in enumerate(letter_imgs[letter]):
            index = n + (class_ - 1) * sample_size
            data[index, :] = image.reshape(1, dim)
            labels[index] = class_
    return data[1:sample_size, :], labels[1:sample_size, :]



def demo():
    dir_dataset = '../../resource/dataset/fingerspelling5/dataset5/'
    n_data = 10
    data, labels = gendata_sign(getpaths_asl(dir_dataset), n_data)
    print("Data:\n")
    print(data)
    print("Labels:\n")
    print(labels)

# demo()
