import cv2
import imutils
from numpy.ma import floor

from classification.models.ColourModelVideo import ColourModelVideo
from classification.models.ShapeModel import ShapeModel
from daq.cam.InputGenerator import InputGenerator
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

            inputgen = InputGenerator(0.5)
            colour_model = ColourModelVideo()
            shape_model = ShapeModel('../../../resource/models/model.pkl')
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

                # convert the roi to grayscale and blur it
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                if gray is None:
                    print("gray is null")
                    return
                gray = cv2.GaussianBlur(gray, (7, 7), 0)

                # to get the background, keep looking till a threshold is reached
                # so that our running average model gets calibrated
                if self.num_frames < 30:
                    inputgen.run_avg(gray)
                elif not colour_model.trained:
                    if (self.num_frames - 30) % 100 == 0:
                        print("Put your hand in the green box and press 'c' to calibrate.")
                    if keypress == ord("c"):
                        colour_model.train(inputgen.background, roi)
                        print("Calibrated ..")
                elif (self.num_frames - 30) % 5 == 0:
                    # segment the hand region
                    hand = colour_model.segment(roi)

                    hand_gray = cv2.cvtColor(hand, cv2.COLOR_RGB2GRAY)
                    hand_gray = cv2.resize(hand_gray, (30, 30))
                    cv2.imshow("Segmented Hand", hand_gray)

                    shape_model.stack_descr(extract_descriptor(hand_gray))
                    if (self.num_frames - 30) % 15 == 0:
                        # every X frames classify and apply majority vote
                        letter = shape_model.predict()
                        print("Detected Letter " + letter)

                # draw the segmented hand
                cv2.rectangle(clone, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom), (0, 255, 0), 2)

                # display the frame with segmented hand
                cv2.imshow("Video Feed", clone)

                # increment the number of frames
                self.num_frames += 1

                # if the user pressed "q", then stop looping
                if keypress == ord("q"):
                    break

            # free up memory
            self.camera.release()
            cv2.destroyAllWindows()


inputhandler = InputHandler()
inputhandler.run()
