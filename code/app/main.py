from app.FrameHandler import FrameHandler
from app.UserInterface import UserInterface

frame_handler = FrameHandler("../../resource/models/model.pkl")

ui = UserInterface(frame_handler)

frame_handler.daemon = True
frame_handler.start()
ui.run()
