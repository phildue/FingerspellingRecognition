from abc import abstractmethod


class Segmenter:
    @abstractmethod
    def get_label(self, img):
        pass

    @abstractmethod
    def get_label_soft(self, img):
        pass
