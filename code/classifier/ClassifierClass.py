from sklearn import svm

from exceptions.NotTrained import NotTrained


class Classifier:

    scikit_object = svm.SVC()
    trained = False

    def __init__(self):
        self.scikit_object = svm.SVC()

    def train(self, data, labels):
        self.scikit_object.fit(data, labels.ravel())
        self.trained = True

    def predict(self, data):
        if not self.trained:
            raise NotTrained
        else:
            return self.scikit_object.predict(data)



