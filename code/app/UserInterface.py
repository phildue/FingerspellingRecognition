import cv2
import imutils
from numpy.ma import floor

from app.FrameHandler import FrameHandler

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]


class UserInterface:
    def __init__(self, frame_handler: FrameHandler):
        self.num_frames = 0
        self.camera = None
        self.roi_left = 0
        self.roi_right = 0
        self.roi_top = 0
        self.roi_bottom = 0
        self.frame_handler = frame_handler

    def set_roi(self, l, r, t, b):
        self.roi_left = l
        self.roi_right = r
        self.roi_top = t
        self.roi_bottom = b

    def run(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            # keep looping, until interrupted
            keypress = cv2.waitKey(1) & 0xFF
            calibrate_print_flag = calibrated_print_flag = False
            while keypress != ord("q"):
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

                # draw roi
                cv2.rectangle(clone, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom), (0, 255, 0),
                              2)

                roi = frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]

                if roi is None:
                    print("roi is null")
                    return
                # display the frame
                cv2.imshow("Video Feed", clone)

                # pass frame to frame handler
                self.frame_handler.add_frame(roi)

                if self.frame_handler.is_ready_to_calibrate() and not calibrate_print_flag:
                    print("Put your hand in the green box and press c to calibrate...")
                    calibrate_print_flag = True
                if self.frame_handler.is_calibrated() and not calibrated_print_flag:
                    print("Calibrated..")
                    calibrated_print_flag = True

                if keypress == ord('c') and \
                        self.frame_handler.is_ready_to_calibrate() and not self.frame_handler.is_calibrated():
                    self.frame_handler.start_calibration()

                # print letter
                if self.frame_handler.is_calibrated():
                    letter = self.frame_handler.get_letter()
                    if letter is None:
                        print("No letter recognized")
                    else:
                        print("Recognized letter: " + letter)

            # free up memory
            self.camera.release()
            self.frame_handler.stop_()
            cv2.destroyAllWindows()
