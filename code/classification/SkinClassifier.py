from __builtin__ import file
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

from daq.datageneration.DatasetGeneratorSkin import gendata_skin
from exceptions.NotTrained import NotTrained


class SkinClassifier:
    scikit_object = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
    trained = False

    def __init__(self, trainedc_path='trained_skin_classifier.pkl'):
        if file.exists(trainedc_path):
            self.scikit_object = joblib.load(trainedc_path)
        else:
            data, labels = gendata_skin()
            self.train(data, labels)
            joblib.dump(self.scikit_object, trainedc_path)

    def train(self, data, labels):
        self.scikit_object.fit(data, labels.ravel())
        self.trained = True

    def predict(self, data):
        if not self.trained:
            raise NotTrained
        else:
            return self.scikit_object.predict(data)
