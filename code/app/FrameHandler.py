import threading
from multiprocessing import Queue

import cv2

from app.models.BackgroundSubtractor import InputGenerator
from app.models.HogEstimator import HogEstimator
from app.models.MrfSegmenter import MRFSegmenter
from preprocessing.preprocessing_asl import extract_descriptor


class FrameHandler(threading.Thread):
    frame_queue = Queue()
    calibrate = threading.Event()
    ready_to_calibrate = threading.Event()
    calibrated = threading.Event()
    stop_ = threading.Event()
    s_letter = threading.Lock()
    detected_letter = None

    def __init__(self, hog_model_path):
        threading.Thread.__init__(self)
        self.inputgen = InputGenerator(0.5)
        self.colour_model = MRFSegmenter()
        self.shape_model = HogEstimator(hog_model_path)

    def run(self):

        num_frames = 0
        while not self.stop_.is_set():
            frame = self.frame_queue.get(True)
            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our running average model gets calibrated
            if not self.colour_model.trained:
                if num_frames < 30:
                    self.inputgen.run_avg(gray)
                elif not self.colour_model.trained:
                    self.ready_to_calibrate.set()
                    self.calibrate.wait()
                    self.colour_model.train(self.inputgen.background, frame)
                    self.calibrated.set()
            else:
                # segment the hand region
                hand = self.colour_model.segment(frame)

                hand_gray = cv2.cvtColor(hand, cv2.COLOR_RGB2GRAY)
                hand_gray = cv2.resize(hand_gray, (30, 30))
                cv2.imshow("Segmented Hand", hand_gray)

                self.shape_model.stack_descr(extract_descriptor(hand_gray))
                if (num_frames - 30) % 15 == 0:
                    # every X frames classify and apply majority vote
                    letter = self.shape_model.predict()
                    self.s_letter.acquire()
                    self.detected_letter = letter
                    self.s_letter.release()
            # increment the number of frames
            num_frames += 1

    def start_calibration(self):
        self.calibrate.set()

    def stop(self):
        self.stop_.set()

    def get_letter(self):
        if self.s_letter.acquire(False):
            letter = self.detected_letter
            self.s_letter.release()
            return letter
        else:
            return None

    def is_ready_to_calibrate(self):
        return self.ready_to_calibrate.is_set()

    def is_calibrated(self):
        return self.calibrated.is_set()

    def add_frame(self, frame):
        self.frame_queue.put(frame, block=False)
