import numpy as np
from daq.imreader import read_letters

from daq.dataset.preprocessing import extract_descriptors, preprocesss


def gendata_sign(img_file_paths,
                 sample_size=2500, letters=None):
    if letters is None:
        letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                   "u",
                   "v", "w", "x", "y"]
    img_lists = read_letters(img_file_paths, sample_size, letters)

    img_pp_dict = dict((letter, preprocesss(img_lists[letter])) for letter in img_lists)

    descriptor_dict = dict((letter,extract_descriptors(img_pp_dict[letter])) for letter in img_pp_dict)

    dim = next(iter(descriptor_dict.values()))[0].size
    sample_size_valid = 0
    for k in descriptor_dict.keys():
        sample_size_valid += len(descriptor_dict[k])

    print("Dimension: " + str(dim), "Samples: " + str(sample_size_valid))

    return vectorize(descriptor_dict,
                     dim=dim,
                     sample_size=sample_size_valid)


def vectorize(descriptor_lists, dim, sample_size):
    data = np.zeros(shape=(sample_size, dim), dtype=np.uint8)
    labels = np.zeros(shape=(sample_size, 1), dtype=np.uint8)
    index = 0
    for class_, letter in enumerate(sorted(descriptor_lists.keys()), 1):
        for n, descriptor in enumerate(descriptor_lists[letter]):
            data[index, :] = descriptor
            labels[index] = class_
            index += 1

    return data, labels


