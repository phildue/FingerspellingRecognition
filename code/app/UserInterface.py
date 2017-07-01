import cv2
import imutils
from numpy.ma import floor

from app.FrameHandler import FrameHandler


class UserInterface:
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
               "u",
               "v", "w", "x", "y"]

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
            keypress = cv2.waitKey(20) & 0xFF
            calibrate_print_flag = calibrated_print_flag = False
            letter_last = None
            (grabbed, frame) = self.camera.read()
            frame = self.format_frame(frame)
            (height, width) = frame.shape[:2]
            # get the ROI
            self.set_roi(int(floor(1 / 2 * width)), int(floor(4 / 4 * width)), int(floor(0 / 3 * height)),
                         int(floor(1 / 2 * height)))

            while keypress != ord("q"):
                (grabbed, frame) = self.camera.read()
                frame = self.format_frame(frame)

                keypress = cv2.waitKey(50) & 0xFF

                frame = self.draw_roi(frame)

                cv2.imshow("Video Feed", frame)

                # pass roi to frame handler
                roi = self.get_roi(frame)
                roi = cv2.flip(roi, 1)

                self.frame_handler.add_frame(roi)

                # print letter
                if self.frame_handler.is_calibrated():
                    letter = self.frame_handler.get_letter()
                    if letter is not letter_last:
                        if letter is -1:
                            print("No letter recognized")
                        elif letter is not None:
                            print("Recognized letter: " + self.letters[letter])
                        letter_last = letter
                else:
                    if self.frame_handler.is_ready_to_calibrate():
                        if keypress == ord('c'):
                            self.frame_handler.start_calibration(roi)
                        if not calibrate_print_flag:
                            print("Put your hand in the green box and press c to calibrate...")
                            calibrate_print_flag = True
                    if self.frame_handler.is_calibrated() and not calibrated_print_flag:
                        print("Calibrated..")
                        calibrated_print_flag = True

            # free up memory
            self.camera.release()
            self.frame_handler.stop()
            cv2.destroyAllWindows()

    def format_frame(self, frame):
        frame = imutils.resize(frame, width=800)
        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)
        # clone the frame
        return frame.copy()

    def get_roi(self, frame):
        return frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]

    def draw_roi(self, frame):
        return cv2.rectangle(frame, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom), (0, 255, 0),
                             2)
