from app.FrameHandler import FrameHandler
from app.UserInterface import UserInterface
from classification.EstimatorVideo import EstimatorVideo
from preprocessing.PreProcessorVideo import PreProcessorVideo
from preprocessing.representation.HistogramOfGradients import HistogramOfGradients
from preprocessing.segmentation.BackgroundSubtractor import BackgroundSubtractor
from preprocessing.segmentation.MRFVideo import MRFVideo

preprocessor = PreProcessorVideo(BackgroundSubtractor(0.5), segmenter=MRFVideo(),
                                 descriptor=HistogramOfGradients(window_size=6, n_bins=8))

frame_handler = FrameHandler(preprocessor, EstimatorVideo("../../resource/models/model_hog_asl.pkl"))

ui = UserInterface(frame_handler)

frame_handler.daemon = True
frame_handler.start()
ui.run()
