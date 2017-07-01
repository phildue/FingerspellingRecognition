import numpy as np

from preprocessing.representation.Descriptor import Descriptor


class Pixel(Descriptor):
    def get_descr(self, img: np.array):
        return img.reshape(1. - 1)
