import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from exceptions.exceptions import NotTrained


class SignClassifier:
    scikit_object = SVC()
    trained = False

    def __init__(self, gamma=None, C=None):
        if gamma is not None and C is not None:
            self.scikit_object = SVC(gamma=gamma, C=C)
        elif gamma is not None:
            self.scikit_object = SVC(gamma=gamma)
        elif C is not None:
            self.scikit_object = SVC(C)

    def train(self, data, labels):
        self.scikit_object.fit(data, labels.ravel())
        self.trained = True

    def predict(self, data):
        if not self.trained:
            raise NotTrained
        else:
            return self.scikit_object.predict(data)

    @staticmethod
    def classify(descriptor: np.array, classifier_file: str) -> int:
        clf = pickle.load(classifier_file)
        return clf.predict(descriptor)
