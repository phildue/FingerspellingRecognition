import cv2
import imutils
from sklearn.externals import joblib

from classification.ColourModel import ColourModel
from daq.cam.InputGenerator import InputGenerator
from daq.dataset.preprocessing import extract_descriptor

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
            colour_model = ColourModel()

            model = joblib.load('../../../resource/models/model.pkl')

            # keep looping, until interrupted
            while True:
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
                self.set_roi(width // 2, width, 0, 2 * height // 3)

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

                edges = None

                # to get the background, keep looking till a threshold is reached
                # so that our running average model gets calibrated
                if self.num_frames < 30:
                    inputgen.run_avg(gray)
                elif not colour_model.trained:
                    colour_model.train(inputgen.background, roi)
                    print("Calibrated ..")
                elif (self.num_frames - 30) % 50 == 0:
                    # segment the hand region
                    hand = colour_model.segment(roi)
                    hand_gray = cv2.cvtColor(hand, cv2.COLOR_RGB2GRAY)
                    # extract descriptor
                    descriptor = extract_descriptor(hand_gray)
                    # classify
                    class_ = model.predict(descriptor)
                    # print output
                    print("Detected Letter " + str(letters[int(class_) - 1]))
                    print("Actual Letter: " + 'a')

                # draw the segmented hand
                cv2.rectangle(clone, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom), (0, 255, 0), 2)

                # display the frame with segmented hand
                cv2.imshow("Video Feed", clone)


                # increment the number of frames
                self.num_frames += 1

                # observe the keypress by the user
                keypress = cv2.waitKey(1) & 0xFF

                # if the user pressed "q", then stop looping
                if keypress == ord("q"):
                    break

            # free up memory
            self.camera.release()
            cv2.destroyAllWindows()


inputhandler = InputHandler()
inputhandler.run()
