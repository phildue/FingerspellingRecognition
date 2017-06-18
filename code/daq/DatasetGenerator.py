import numpy as np

from daq.ImReader import read_letters
from daq.preprocessing.PreProcessing import extract_descriptors


def gendata_sign(img_file_paths,
                 sample_size=2500, letters=None):
    if letters is None:
        letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                   "u",
                   "v", "w", "x", "y"]
    img_lists = read_letters(img_file_paths, sample_size, letters)

    descriptor_lists = {}
    for letter in img_lists:
        descriptor_lists[letter] = extract_descriptors(img_lists[letter])

    dim = len(next(iter(descriptor_lists.values()))[0])*2

    return vectorize(descriptor_lists,
                     dim=dim,
                     sample_size=sample_size)


def vectorize(descriptor_lists, dim, sample_size):
    n_letters = len(descriptor_lists)
    data = np.zeros(shape=(sample_size * n_letters, dim), dtype=np.uint8)
    labels = np.zeros(shape=(sample_size * n_letters, 1), dtype=np.uint8)
    for class_, letter in enumerate(sorted(descriptor_lists.keys()), 1):
        for n, image in enumerate(descriptor_lists[letter]):
            index = n + (class_ - 1) * sample_size
            data[index, :] = image.reshape(1, dim)
            labels[index] = class_
    return data, labels


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


def demo():
    dir_dataset = '../../resource/dataset/fingerspelling5/dataset5/'
    n_data = 10
    data, labels = gendata_sign(getpaths_asl(dir_dataset), n_data)
    print("Data:\n")
    print(data)
    print("Labels:\n")
    print(labels)

# demo()
