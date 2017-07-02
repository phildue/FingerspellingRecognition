import cv2
import numpy as np

from preprocessing.PreProcessor import PreProcessor
from preprocessing.representation.Descriptor import Descriptor
from preprocessing.representation.HistogramOfGradients import HistogramOfGradients
from preprocessing.segmentation.MRFAsl import MRFAsl
from preprocessing.segmentation.Segmenter import Segmenter


class PreProcessorAsl(PreProcessor):
    def __init__(self, descriptor: Descriptor = HistogramOfGradients(), segmenter: Segmenter = MRFAsl(),
                 img_size=(60, 60)):
        self.img_size = img_size
        self.descriptor = descriptor
        self.segmenter = segmenter

    def get_descr(self, img):
        return self.descriptor.get_descr(img).reshape(1, -1)

    def preprocess(self, img):
        label_map = self.segmenter.get_label(img)

        img_extracted = img[0].copy()
        img_extracted[label_map == 0] = (0, 0, 0)

        img_extracted = cv2.cvtColor(img_extracted, cv2.COLOR_RGB2GRAY)
        img_extracted = cv2.resize(img_extracted, self.img_size)
        return cv2.equalizeHist(img_extracted)
