import cv2
import imutils

from daq.cam.InputGenerator import InputGenerator


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
            # keep looping, until interrupted
            while True:
                # get the current frame
                (grabbed, frame) = self.camera.read()

                # resize the frame
                frame = imutils.resize(frame, width=700)

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

                # to get the background, keep looking till a threshold is reached
                # so that our running average model gets calibrated
                if self.num_frames < 30:
                    inputgen.run_avg(gray)
                else:
                    # segment the hand region
                    hand = inputgen.get_foreground(gray)

                    # check whether hand region is segmented
                    if hand is not None:
                        # if yes, unpack the thresholded image and
                        # segmented region
                        (edges, segmented) = hand

                        # draw the segmented region and display the frame
                        cv2.drawContours(clone, [segmented + (self.roi_right, self.roi_top)], -1, (0, 0, 255))
                        cv2.imshow("Hand Contour", edges)

                # draw the segmented hand
                cv2.rectangle(clone, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom), (0, 255, 0), 2)

                # increment the number of frames
                self.num_frames += 1

                # display the frame with segmented hand
                cv2.imshow("Video Feed", clone)

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
