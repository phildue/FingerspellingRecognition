import os

import cv2
import numpy as np


def gendata(dir_dataset, n_data):
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
                "w", "x", "y"]
    n_letters = 24
    im_res = (100, 120, 3)
    dim = im_res[0] * im_res[1] * im_res[2]
    data = np.zeros(shape=(n_data * n_letters, dim))
    labels = np.zeros(shape=(n_data * n_letters, 1))
    for class_, directory in enumerate(alphabet, 1):
        for n_object, filename in enumerate(os.listdir(dir_dataset + directory)):
            if n_object >= n_data:
                break
            img = cv2.imread(dir_dataset + directory + "/" + filename, 1)
            img = cv2.resize(img, (im_res[0], im_res[1]))
            index = n_object + (class_ - 1) * n_data
            data[index, :] = img.reshape(1, dim)
            labels[index] = class_


    return data, labels


def main():
    dir_dataset = '../../resource/dataset/fingerspelling5/dataset5/A/'
    n_data = 10
    data, labels = gendata(dir_dataset, n_data)
    print("Data:\n")
    print(data)
    print("Labels:\n")
    print(labels)


#main()
