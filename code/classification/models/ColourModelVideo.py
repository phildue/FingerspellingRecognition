import cv2
import numpy as np
from skimage.segmentation import slic
from sklearn.neighbors import KNeighborsClassifier

from exceptions.exceptions import NotTrained


class ColourModelVideo:
    model = None
    trained = False

    def train(self, background, image):

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(background.astype("uint8"), gray)
        cv2.imshow("Roi", diff)
        _, binary = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        roi = image[binary == 255].reshape(-1, 3)
        negative = image[binary == 0].reshape(-1, 3)
        test = image.copy()
        test[binary == 0] = (0, 0, 0)
        cv2.imshow("Roi", test)
        roi_label = np.zeros(shape=(roi.shape[0], 1))
        negative_label = np.ones(shape=(negative.shape[0], 1))
        data = np.vstack((roi, negative))
        labels = np.vstack((roi_label, negative_label))
        self.model = KNeighborsClassifier(3)
        self.model.fit(data, labels.ravel())
        self.trained = True

    def segment(self, image):
        if not self.trained:
            raise NotTrained
        superpixels = slic(image, n_segments=800)
        assignments = np.zeros(shape=image.shape[0:2])
        for i in range(0, np.max(superpixels)):
            if len(image[superpixels == i]) > 0:
                mean_sp = np.mean(image[superpixels == i], axis=0)
                assignments[superpixels == i] = self.model.predict(mean_sp.reshape(1, -1))

        segmented = image.copy()
        segmented[assignments == 1] = (40, 40, 40)
        return segmented
