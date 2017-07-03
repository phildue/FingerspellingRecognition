import threading
from multiprocessing import Queue

from classification.EstimatorVideo import EstimatorVideo
from preprocessing.PreProcessorVideo import PreProcessorVideo


class FrameHandler(threading.Thread):
    frame_queue = Queue()
    ready_to_calibrate = threading.Event()
    calibrated = threading.Event()
    stop_ = threading.Event()
    s_letter = threading.Lock()
    detected_letter = None

    def __init__(self, preprocessor: PreProcessorVideo, estimator: EstimatorVideo):
        threading.Thread.__init__(self)
        self.preprocessor = preprocessor
        self.estimator = estimator

    def run(self):

        num_frames = 0
        while not self.stop_.is_set():
            frame = self.frame_queue.get(True)

            if self.calibrated.is_set():
                # segment the hand region
                hand = self.preprocessor.preprocess(frame)

                self.estimator.stack_descr(self.preprocessor.get_descr(hand))
                if num_frames % 5 == 0:
                    # every X frames classify and apply majority vote
                    self.s_letter.acquire()
                    self.detected_letter = self.estimator.predict()
                    self.s_letter.release()
            else:
                if num_frames < 30:
                    self.preprocessor.calibrate_background(frame)
                else:
                    self.ready_to_calibrate.set()
                    self.calibrated.wait()
                    num_frames = 0

            num_frames += 1

    def start_calibration(self, frame):
        self.preprocessor.calibrate_object(frame)
        self.calibrated.set()
        self.ready_to_calibrate.clear()

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
