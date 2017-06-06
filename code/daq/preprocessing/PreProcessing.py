import cv2
from numpy import shape

from daq.preprocessing.SkinSegmentor import filter_skin


class PreProcessor:
    im_size = shape

    def __init__(self, im_size):
        self.im_size = im_size

    def apply_pp(self, img):
        img_preprocessed = cv2.resize(img, (self.im_size[0], self.im_size[1]))
        img_preprocessed = filter_skin(img_preprocessed)
        img_preprocessed = cv2.cvtColor(img_preprocessed, cv2.COLOR_RGB2GRAY)
        return img_preprocessed
