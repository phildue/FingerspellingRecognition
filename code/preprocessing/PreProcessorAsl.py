import cv2
import numpy as np

from preprocessing.PreProcessor import PreProcessor
from preprocessing.representation.Descriptor import Descriptor
from preprocessing.segmentation.Segmenter import Segmenter


class PreProcessorAsl(PreProcessor):
    def __init__(self, descriptor: Descriptor, segmenter: Segmenter, img_size=(60, 60)):
        self.img_size = img_size
        self.descriptor = descriptor
        self.segmenter = segmenter

    def get_descr(self, img):
        return self.descriptor.get_descr(img)

    def preprocess(self, img):
        img_segment = self.segmenter.get_label(img)

        img_segment = cv2.cvtColor(img_segment, cv2.COLOR_RGB2GRAY)
        img_segment = cv2.resize(img_segment, self.img_size)
        return cv2.equalizeHist(img_segment)
