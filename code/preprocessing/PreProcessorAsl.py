import cv2
import numpy as np

from preprocessing.PreProcessor import PreProcessor
from preprocessing.representation.Descriptor import Descriptor
from preprocessing.segmentation.Segmenter import Segmenter


class PreProcessorAsl(PreProcessor):
    def __init__(self, descriptor: Descriptor, segmenter: Segmenter, img_size=(60, 60), confidence_thresh=250):
        self.img_size = img_size
        self.confidence_thresh = confidence_thresh
        self.descriptor = descriptor
        self.segmenter = segmenter

    def get_descr(self, img):
        return self.descriptor.get_descr(img)

    def preprocess(self, img):
        label_map = cv2.normalize(self.segmenter.get_label().astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imshow("Labels", label_map)
        label_map = cv2.GaussianBlur(label_map, (3, 3), 2)

        kernel = np.ones((5, 5), np.uint8)

        label_map = cv2.morphologyEx(label_map, cv2.MORPH_CLOSE, kernel, iterations=5)

        _, label_map = cv2.threshold(label_map, self.confidence_thresh, 255, cv2.THRESH_BINARY)

        img_extracted = img.copy()
        img_extracted[label_map == 0] = 0

        img_segment = cv2.cvtColor(img_extracted, cv2.COLOR_RGB2GRAY)
        img_segment = cv2.resize(img_segment, self.img_size)
        return cv2.equalizeHist(img_segment)
