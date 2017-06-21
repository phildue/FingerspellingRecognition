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

    def predict(self, frames: [np.array]):
        votes = np.zeros(shape=(len(self.letters)))
        for frame in frames:
            votes[self.model.predict(frame) - 1] += 1

        return self.letters[np.argmax(votes)]
