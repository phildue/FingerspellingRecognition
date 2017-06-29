import numpy as np
from sklearn.externals import joblib


class HogEstimator:
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
               "u",
               "v", "w", "x", "y"]
    model = None
    trained = False
    iterations = 0

    def __init__(self, model_path='../../resource/models/model.pkl'):
        self.model = joblib.load(model_path)
        self.trained = True
        self.votes = np.zeros(shape=(len(self.letters)))

    def predict(self, ):
        if np.max(self.votes) > self.iterations * 0.75:
            class_ = self.letters[int(np.argmax(self.votes))]
            self.votes = np.zeros(shape=(len(self.letters)))
            self.iterations = 0
            return class_
        else:
            return "No letter recognized"

    def stack_descr(self, descriptor):
        self.votes[self.model.predict(descriptor.astype(np.uint8)) - 1] += 1
        self.iterations += 1
