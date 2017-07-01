from abc import abstractmethod

import numpy as np


class Descriptor:
    @abstractmethod
    def get_descr(self, img: np.array):
        pass

    def get_descriptors(self, imgs):
        return [self.get_descr(img) for img in imgs]
