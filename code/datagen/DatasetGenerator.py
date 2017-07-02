import numpy as np
from sklearn.externals import joblib

from datagen.FileProvider import FileProvider
from preprocessing.PreProcessor import PreProcessor


class DatasetGenerator:
    def __init__(self, file_provider: FileProvider, preprocessor: PreProcessor, letters=None):

        self.preprocessor = preprocessor
        self.file_provider = file_provider

        if letters is None:
            self.letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                            "t",
                            "u",
                            "v", "w", "x", "y"]

    @staticmethod
    def load(sample_file_path, label_file_path):
        data = joblib.load(sample_file_path)
        labels = joblib.load(label_file_path)

        print("Loaded Samples: " + str(data.shape[0]))

        return data, labels

    @staticmethod
    def save(data, labels, path: str, name: str):
        joblib.dump(data, path + 'descriptors_' + name + '.pkl')
        joblib.dump(labels, path + 'labels_' + name + '.pkl')

    def generate(self, sample_size=2500):

        img_lists = self.file_provider.read_letters(sample_size, self.letters)

        img_pp_dict = dict((letter, self.preprocessor.preprocess_all(img_lists[letter])) for letter in self.letters)

        descriptor_dict = dict(
            (letter, self.preprocessor.get_descr_all(img_pp_dict[letter])) for letter in self.letters)

        dim = next(iter(descriptor_dict.values()))[0].size
        sample_size_valid = 0
        for letter in self.letters:
            sample_size_valid += len(descriptor_dict[letter])

        print("Dimension: " + str(dim), "Samples: " + str(sample_size_valid))

        return self.vectorize(descriptor_dict,
                              dim=dim,
                              sample_size=sample_size_valid)

    @staticmethod
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
