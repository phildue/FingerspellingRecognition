from sklearn.svm import SVC

from exceptions.exceptions import NotTrained


class SignClassifier:

    scikit_object = SVC()
    trained = False

    def train(self, data, labels):
        self.scikit_object.fit(data, labels.ravel())
        self.trained = True

    def predict(self, data):
        if not self.trained:
            raise NotTrained
        else:
            return self.scikit_object.predict(data)



