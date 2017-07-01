import cv2
import maxflow as mf
import numpy as np

from preprocessing.segmentation.MarkovRandomField import MarkovRandomField
from preprocessing.segmentation.Segmenter import Segmenter


class MRFVideo(MarkovRandomField):
    def get_soft_labelling(self):
        pass

    def get_label(self):
        pass
