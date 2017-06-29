import threading

from app.FrameHandler import FrameHandler
from app.UserInterface import UserInterface

frame_handler = FrameHandler()

frame_handler_thread = threading.Thread(target=frame_handler.run(), args=())

ui = UserInterface(frame_handler)

frame_handler_thread.daemon = True
frame_handler_thread.start()

ui.run()
