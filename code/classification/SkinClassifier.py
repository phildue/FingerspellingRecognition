from sklearn.neighbors import KNeighborsClassifier

from exceptions.NotTrained import NotTrained


class SkinClassifier:

    scikit_object = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
    trained = False

    def train(self, data, labels):
        self.scikit_object.fit(data, labels.ravel())
        self.trained = True

    def predict(self, data):
        if not self.trained:
            raise NotTrained
        else:
            return self.scikit_object.predict(data)



