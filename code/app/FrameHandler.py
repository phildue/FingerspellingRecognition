import threading
from multiprocessing import Queue

import cv2

from app.models.PreprocessorVideo import PreprocessorVideo
from app.models.EstimatorVideo import EstimatorVideo
from preprocessing.preprocessing_asl import extract_descriptor


class FrameHandler(threading.Thread):
    frame_queue = Queue()
    calibrate = threading.Event()
    ready_to_calibrate = threading.Event()
    calibrated = threading.Event()
    stop_ = threading.Event()
    s_letter = threading.Lock()
    detected_letter = None

    def __init__(self, preprocessor: PreprocessorVideo, estimator: EstimatorVideo):
        threading.Thread.__init__(self)
        self.preprocessor = preprocessor
        self.estimator = estimator

    def run(self):

        num_frames = 0
        while not self.stop_.is_set():
            frame = self.frame_queue.get(True)
            # convert the roi to grayscale and blur it

            # to get the background, keep looking till a threshold is reached
            # so that our running average model gets calibrated
            if not self.preprocessor.calibrated:
                if num_frames < 30:
                    self.preprocessor.calibrate_background(frame)
                else:
                    self.ready_to_calibrate.set()
                    self.calibrate.wait()
                    self.preprocessor.calibrate_object(frame)
                    self.calibrated.set()
            else:
                # segment the hand region
                hand = self.preprocessor.preprocess(frame)

                cv2.imshow("Segmented Hand", hand)
                cv2.waitKey(10)
                self.estimator.stack_descr(extract_descriptor(hand))
                if (num_frames - 30) % 15 == 0:
                    # every X frames classify and apply majority vote
                    self.s_letter.acquire()
                    self.detected_letter = self.estimator.predict()
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
