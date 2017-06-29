import cv2
import imutils
from classification.models.MrfSegmenter import MRFSegmenter
from numpy.ma import floor

from app.models.BackgroundSubtractor import InputGenerator
from app.models.HogEstimator import HogEstimator
from preprocessing.segmentation_tm import extract_descriptor

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]


class InputHandler:
    def __init__(self):
        self.num_frames = 0
        self.camera = None
        self.roi_left = 0
        self.roi_right = 0
        self.roi_top = 0
        self.roi_bottom = 0

    def set_roi(self, l, r, t, b):
        self.roi_left = l
        self.roi_right = r
        self.roi_top = t
        self.roi_bottom = b

    def run(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            self.num_frames = 0

            # keep looping, until interrupted
            while True:
                # observe the keypress by the user
                keypress = cv2.waitKey(1) & 0xFF
                # get the current frame
                (grabbed, frame) = self.camera.read()

                # resize the frame
                frame = imutils.resize(frame, width=800)

                # flip the frame so that it is not the mirror view
                frame = cv2.flip(frame, 1)

                # clone the frame
                clone = frame.copy()
                (height, width) = frame.shape[:2]

                # get the ROI
                self.set_roi(int(floor(1 / 2 * width)), int(floor(3 / 4 * width)), int(floor(1 / 3 * height)),
                             int(floor(2 / 3 * height)))

                roi = frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]

                if roi is None:
                    print("roi is null")
                    return

                # pass frame to frame handler


                if keypress == ord("c"):
                    print("Calibrated ..")

                # draw the segmented hand
                cv2.rectangle(clone, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom), (0, 255, 0),
                              2)

                # display the frame with segmented hand
                cv2.imshow("Video Feed", clone)

                # if the user pressed "q", then stop looping
                if keypress == ord("q"):
                    break

            # free up memory
            self.camera.release()
            cv2.destroyAllWindows()


inputhandler = InputHandler()
inputhandler.run()
