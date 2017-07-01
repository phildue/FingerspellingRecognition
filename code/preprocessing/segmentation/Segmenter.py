from abc import abstractmethod


class Segmenter:
    @abstractmethod
    def get_label(self):
        pass

    @abstractmethod
    def get_label_soft(self):
        pass
