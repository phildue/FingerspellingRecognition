import numpy as np
from sklearn.externals import joblib


class ShapeModel:
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
               "u",
               "v", "w", "x", "y"]
    model = None
    trained = False

    def __init__(self, model_path='../../resource/models/model.pkl'):
        self.model = joblib.load(model_path)
        self.trained = True
        self.votes = np.zeros(shape=(len(self.letters)))

    def predict(self, ):
        values, counts = np.unique(self.votes, return_counts=True)

        class_ = self.letters[int(values[int(np.argmax(counts))])]
        self.votes = np.zeros(shape=(len(self.letters)))

        return class_

    def stack_descr(self, descriptor):
        self.votes[self.model.predict(descriptor) - 1] += 1
