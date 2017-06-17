import os
import random

import numpy as np

from daq.ImReader import read_im_file
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
                 sample_size=2500, alphabet=None, im_size=(100, 120, 3)):
    if alphabet is None:
        alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                    "v", "w", "x", "y"]

    random.shuffle(alphabet)
    n_letters = len(alphabet)
    dim = im_size[0] * im_size[1]  # * im_size[2]
    data = np.zeros(shape=(sample_size * n_letters, dim), dtype=np.uint8)
    labels = np.zeros(shape=(sample_size * n_letters, 1), dtype=np.uint8)

    for class_, letter in enumerate(alphabet, 1):

        try:
            path_sel = random.sample(img_file_paths[letter], sample_size)
        except ValueError:
            print("Too many samples requested for [" + letter + "], taking all: " + str(len(path_sel)))

        for sel_samples, path in enumerate(path_sel):
            img = read_im_file(path)
            img = pre_processing(img, im_size)
            index = sel_samples + (class_ - 1) * sample_size
            data[index, :] = img.reshape(1, dim)
            labels[index] = class_

    return data[1:sample_size, :], labels[1:sample_size, :]


def readdata_asl(dir_dataset='../../resource/dataset/fingerspelling5/dataset5/',
                 sets=None, ):
    if sets is None:
        sets = ["A", "B", "C", "D", "E"]

    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                "v", "w", "x", "y"]
    paths = {}

    for dir_letter in alphabet:
        paths[dir_letter] = [str]
        for dir_set in sets:
            fnames = [f for f in os.listdir(dir_dataset + dir_set + "/" + dir_letter) if 'color' in f]
            for fname in fnames:
                paths[dir_letter].append(dir_dataset + dir_set + "/" + dir_letter + "/" + fname)

    return paths


def readdata_tm(dir_dataset='../../../resource/dataset/tm'):
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                "v", "w", "x", "y"]
    paths = {}

    for dir_letter in alphabet:
        paths[dir_letter] = [str]
        i = 1
        while os.path.isfile(dir_dataset + '/' + dir_letter + str(i) + '.tif'):
            paths[dir_letter].append(dir_dataset + '/' + dir_letter + str(i) + '.tif')
            i += 1

    return paths


def demo():
    dir_dataset = '../../resource/dataset/fingerspelling5/dataset5/'
    n_data = 10
    data, labels = gendata_sign(readdata_asl(dir_dataset), n_data)
    print("Data:\n")
    print(data)
    print("Labels:\n")
    print(labels)

# demo()
