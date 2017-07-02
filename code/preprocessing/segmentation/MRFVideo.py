import cv2
import maxflow as mf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from exceptions.exceptions import NotTrained
from preprocessing.segmentation.MarkovRandomField import MarkovRandomField
from preprocessing.segmentation.Segmenter import Segmenter


class MRFVideo(MarkovRandomField):
    classifier = None
    trained = False
    background = None

    def train(self, average, image):
        self.background = average

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(average.astype("uint8"), gray)
        # cv2.imshow("Roi", diff)
        _, binary = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        roi = image[binary == 255].reshape(-1, 3)
        negative = image[binary == 0].reshape(-1, 3)
        test = image.copy()
        test[binary == 0] = (0, 0, 0)
        # cv2.imshow("Roi", test)
        roi_label = np.ones(shape=(roi.shape[0], 1))
        negative_label = np.zeros(shape=(negative.shape[0], 1))
        data = np.vstack((roi, negative))
        labels = np.vstack((roi_label, negative_label))
        self.classifier = KNeighborsClassifier(3)
        self.classifier.fit(data, labels.ravel())
        self.trained = True

    def get_label(self, img):
        if not self.trained:
            raise NotTrained

        background_score = self.get_background_score(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), self.background)
        classifier_score = self.classifier.predict_proba(img.reshape(-1, 3)).reshape((img.shape[0], img.shape[1], 2))
        likelihood_grid = self.get_weighted_sum(classifier_score, background_score)
        weight_x, weight_y = self.get_smooth_grid(img)

        graph, nodeids = self.create_graph(img.shape, weight_x, weight_y, likelihood_grid[:, :, 1],
                                           likelihood_grid[:, :, 0])

        return self.maxflow(graph, nodeids)
