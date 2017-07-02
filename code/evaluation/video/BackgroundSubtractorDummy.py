from preprocessing.segmentation.BackgroundSubtractor import BackgroundSubtractor


class BackgroundSubtractorDummy(BackgroundSubtractor):
    def __init__(self, diff_img, test_img):
        super().__init__()
        self.diff_img = diff_img
        self.test_img = test_img

    def get_background(self):
        return self.test_img + self.diff_img
