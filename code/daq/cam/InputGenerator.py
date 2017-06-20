import cv2


class InputGenerator:
    def __init__(self, alpha=0.5):
        self.background = None
        self.alpha = alpha

    def run_avg(self, image):
        # initialize the background
        if self.background is None:
            self.background = image.copy().astype("float")
            return
        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image, self.background, self.alpha)

    def get_foreground(self, image):
        # find the absolute difference between background and current frame
        diff = cv2.absdiff(self.background.astype("uint8"), image)

        # detect edges in the image
        edges = cv2.Canny(diff, threshold1=20, threshold2=80)

        # get the contours in the thresholded image
        (_, contours, _) = cv2.findContours(edges.copy(),
                                            cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_SIMPLE)

        # return None, if no contours detected
        if len(contours) == 0:
            return
        else:
            # based on contour area, get the maximum contour which is the hand
            segmented = max(contours, key=cv2.contourArea)
            return edges, segmented
