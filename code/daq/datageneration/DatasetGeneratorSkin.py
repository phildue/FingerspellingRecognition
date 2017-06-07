import numpy as np


def gendata_skin(path_dataset='../../../resource/dataset/skin/Skin_NonSkin.txt',
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



