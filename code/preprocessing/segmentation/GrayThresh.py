import cv2
import numpy as np

from exceptions.exceptions import NoRoiFound, NoContoursFound
from preprocessing.segmentation.Segmenter import Segmenter


class GrayThresh(Segmenter):
    def __init__(self, im_size=(100, 120)):
        self.im_size = im_size

    def get_label(self, img):
        img_resized = cv2.resize(img, self.im_size)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        _, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

        return img_binary

    def get_label_soft(self, img):
        pass
