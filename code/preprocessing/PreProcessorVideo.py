from preprocessing.PreProcessorAsl import PreProcessorAsl
from preprocessing.representation.Descriptor import Descriptor
from preprocessing.representation.HistogramOfGradients import HistogramOfGradients
from preprocessing.segmentation.BackgroundSubtractor import BackgroundSubtractor
from preprocessing.segmentation.MRFVideo import MRFVideo


class PreProcessorVideo(PreProcessorAsl):
    calibrated = False

    def __init__(self, background_subtractor=BackgroundSubtractor(), segmenter: MRFVideo = MRFVideo(),
                 descriptor: Descriptor = HistogramOfGradients()):
        super().__init__(descriptor, segmenter)
        self.background_subtractor = background_subtractor
        self.segmenter = segmenter

    def calibrate_background(self, frame):
        self.background_subtractor.run_avg(frame)

    def calibrate_object(self, frame):
        self.segmenter.train(self.background_subtractor.get_background(), frame)
        self.calibrated = True
