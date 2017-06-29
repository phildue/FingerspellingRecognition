import threading
from multiprocessing import Queue

import cv2

from app.models.BackgroundSubtractor import InputGenerator
from app.models.HogEstimator import HogEstimator
from app.models.MrfSegmenter import MRFSegmenter
from preprocessing.preprocessing_asl import extract_descriptor


class FrameHandler:
    frame_queue = Queue()
    running = True
    calibrate = ready_to_calibrate = calibrated = False
    detected_letter = None
    s_ready_to_calibrate = s_calibrate = s_calibrated = s_letter = s_running = threading.Lock()

    def run(self):
        inputgen = InputGenerator(0.5)
        colour_model = MRFSegmenter()
        shape_model = HogEstimator('../../../resource/models/model.pkl')
        num_frames = 0
        self.s_running.acquire()
        while self.running:
            self.s_running.release()
            frame = self.frame_queue.get()
            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our running average model gets calibrated
            if not colour_model.trained:
                if num_frames < 30:
                    inputgen.run_avg(gray)
                elif not colour_model.trained:
                    self.s_ready_to_calibrate.release()
                self.s_calibrate.acquire()
                if self.calibrate:
                    self.s_calibrate.release()
                    colour_model.train(inputgen.background, frame)
                    self.s_calibrated.acquire()
                    self.calibrated = True
                    self.s_calibrated.release()
            else:
                # segment the hand region
                hand = colour_model.segment(frame)

                hand_gray = cv2.cvtColor(hand, cv2.COLOR_RGB2GRAY)
                hand_gray = cv2.resize(hand_gray, (30, 30))
                cv2.imshow("Segmented Hand", hand_gray)

                shape_model.stack_descr(extract_descriptor(hand_gray))
                if (num_frames - 30) % 15 == 0:
                    # every X frames classify and apply majority vote
                    letter = shape_model.predict()
                    self.s_letter.acquire()
                    self.detected_letter = letter
                    self.s_letter.release()
            # increment the number of frames
            num_frames += 1
            self.s_running.acquire()

    def start_calibration(self):
        self.s_calibrate.acquire()
        self.calibrate = True
        self.s_calibrate.release()

    def stop(self):
        self.s_running.acquire()
        self.running = False
        self.s_running.release()

    def get_letter(self):
        self.s_letter.acquire()
        letter = self.detected_letter.copy()
        self.s_letter.release()
        return letter

    def is_ready_to_calibrate(self):
        self.s_ready_to_calibrate.acquire()
        flag = self.ready_to_calibrate
        self.s_ready_to_calibrate.release()
        return flag

    def is_calibrated(self):
        self.s_calibrated.acquire()
        flag = self.calibrated
        self.s_calibrated.release()
        return flag

    def add_frame(self, frame):
        self.frame_queue.put(frame)
