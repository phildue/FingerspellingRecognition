import cv2


class BackgroundSubtractor:
    def __init__(self, alpha=0.5):
        self.background = None
        self.alpha = alpha

    def run_avg(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)
        # initialize the background
        if self.background is None:
            self.background = image_blur.copy().astype("float")
            return
        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image_blur, self.background, self.alpha)

    def get_diff(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.absdiff(self.background.astype("uint8"), gray)

    def get_background(self):
        return self.background
