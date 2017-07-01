from app.BackgroundSubtractor import BackgroundSubtractor
from app.EstimatorVideo import EstimatorVideo
from app.FrameHandler import FrameHandler
from app.PreprocessorVideo import PreprocessorVideo
from app.SegmenterVideo import SegmenterVideo
from app.UserInterface import UserInterface

preprocessor = PreprocessorVideo(BackgroundSubtractor(0.5), SegmenterVideo())
frame_handler = FrameHandler(preprocessor, EstimatorVideo("../../resource/models/model_eqhist.pkl"))

ui = UserInterface(frame_handler)

frame_handler.daemon = True
frame_handler.start()
ui.run()
