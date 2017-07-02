from abc import abstractmethod


class PreProcessor:
    @abstractmethod
    def preprocess(self, img):
        pass

    @abstractmethod
    def get_descr(self, img):
        pass

    def get_descr_all(self, imgs):
        return [self.get_descr(img) for img in imgs]

    def preprocess_all(self, imgs):
        return [self.preprocess(img) for img in imgs]
