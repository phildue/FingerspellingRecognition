from app.FrameHandler import FrameHandler
from app.UserInterface import UserInterface
from app.models.BackgroundSubtractor import BackgroundSubtractor
from app.models.EstimatorVideo import EstimatorVideo
from app.models.PreprocessorVideo import PreprocessorVideo
from app.models.SegmenterVideo import SegmenterVideo

preprocessor = PreprocessorVideo(BackgroundSubtractor(0.5), SegmenterVideo())
frame_handler = FrameHandler(preprocessor, EstimatorVideo("../../resource/models/model_asl.pkl"))

ui = UserInterface(frame_handler)

frame_handler.daemon = True
frame_handler.start()
ui.run()
