import numpy as np
from sklearn.externals import joblib

from daq.fileaccess import read_letters
from preprocessing.preprocessing_asl import preprocesss, extract_descriptors


def load_data_sign(sample_file_path, label_file_path):
    data = joblib.load(sample_file_path)
    labels = joblib.load(label_file_path)

    print("Samples: " + str(data.shape[0]))

    return data, labels


def gendata_sign(img_file_paths,
                 sample_size=2500, letters=None):
    if letters is None:
        letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                   "u",
                   "v", "w", "x", "y"]
    img_lists = read_letters(img_file_paths, sample_size, letters)

    img_pp_dict = dict((letter, preprocesss(img_lists[letter])) for letter in letters)

    descriptor_dict = dict((letter, extract_descriptors(img_pp_dict[letter])) for letter in letters)

    dim = next(iter(descriptor_dict.values()))[0].size
    sample_size_valid = 0
    for letter in letters:
        sample_size_valid += len(descriptor_dict[letter])

    print("Dimension: " + str(dim), "Samples: " + str(sample_size_valid))

    return vectorize(descriptor_dict,
                     dim=dim,
                     sample_size=sample_size_valid)


def vectorize(descriptor_lists, dim, sample_size):
    data = np.zeros(shape=(sample_size, dim), dtype=np.float)
    labels = np.zeros(shape=(sample_size, 1), dtype=np.float)
    index = 0
    for class_, letter in enumerate(sorted(descriptor_lists.keys()), 1):
        for n, descriptor in enumerate(descriptor_lists[letter]):
            data[index, :] = descriptor
            labels[index] = class_
            index += 1

    return data, labels
