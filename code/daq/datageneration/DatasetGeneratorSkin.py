import os
import numpy as np

file_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(file_path, os.path.pardir))
path_data = os.path.join(project_path, '../../resource/dataset/skin/Skin_NonSkin.txt')


def gendata_skin(path_dataset=os.path.abspath(path_data), sample_size=245000):
    data = np.zeros(shape=(sample_size, 3))
    labels = np.zeros(shape=(sample_size, 1))

    #print(path_dataset)
    with open(path_dataset) as f:
        for n, line in enumerate(f):
            if n >= sample_size:
                break
            elements = line.split()
            data[n, :] = np.array(list(map(int, elements[0:3])))
            labels[n, 0] = np.array(list(map(int, elements[3])))

    return data[1:sample_size, :], labels[1:sample_size, :]
