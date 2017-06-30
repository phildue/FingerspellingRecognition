import cv2

from preprocessing.preprocessing_asl import extract_descriptor
from preprocessing.segmentation import extract_label


class PreprocessorVideo:
    calibrated = False

    def __init__(self, background_subtractor, segmenter):
        self.background_subtractor = background_subtractor
        self.segmenter = segmenter

    def preprocess(self, frame):
        hand_label = self.segmenter.segment(frame)
        hand = extract_label(frame, hand_label, confidence_thresh=110)
        hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        return cv2.resize(hand, (60, 60))

    def extract_descriptor(self, frame):
        return extract_descriptor(frame)

    def calibrate_background(self, frame):
        self.background_subtractor.run_avg(frame)

    def calibrate_object(self, frame):
        self.segmenter.train(self.background_subtractor.background, frame)
        self.calibrated = True