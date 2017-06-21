import cv2
from numpy import zeros, vstack, ones
from sklearn.neighbors import KNeighborsClassifier

from exceptions.exceptions import NotTrained


class ColourModel:
    model = None
    trained = False

    def train(self, background, image):
        gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(background.astype("uint8"), gray)
        binary = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)
        roi = image[binary == 255].reshape(-1, 3)
        negative = image[binary == 0].reshape(-1, 3)
        roi_label = zeros(shape=(roi.shape[0]))
        negative_label = ones(shape=(negative.shape[0]))
        data = vstack((roi, negative))
        labels = vstack((roi_label, negative_label))
        self.model = KNeighborsClassifier(3)
        self.model.fit(data, labels)
        self.trained = True

    def segment(self, image):
        if not self.trained:
            raise NotTrained

        segmented = image.copy()
        for y in range(0, image.shape[0]):
            for x in range(0, image.shape[1]):
                class_ = self.model.predict(image[y, x, :])
                segmented[y, x, :] = image[y, x, :] if class_ == 1 else (0, 0, 0)

        return segmented
