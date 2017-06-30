from math import floor

import cv2
import imutils

from app.BackgroundSubtractor import BackgroundSubtractor
from app.PreprocessorVideo import PreprocessorVideo
from app.SegmenterVideo import SegmenterVideo

camera = cv2.VideoCapture(0)
# keep looping, until interrupted
keypress = cv2.waitKey(1) & 0xFF
calibrate_print_flag = calibrated_print_flag = False

roi_left = 0
roi_right = 0
roi_top = 0
roi_bottom = 0

preprocessor = PreprocessorVideo(BackgroundSubtractor(0.5), SegmenterVideo())
num_frames = 0
while keypress != ord("q"):
    # observe the keypress by the user
    keypress = cv2.waitKey(10) & 0xFF
    # get the current frame
    (grabbed, frame) = camera.read()

    # resize the frame
    frame = imutils.resize(frame, width=800)
    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # clone the frame
    clone = frame.copy()
    (height, width) = frame.shape[:2]
    # get the ROI
    roi_left = int(floor(1 / 2 * width))
    roi_right = int(floor(3 / 4 * width))
    roi_top = int(floor(1 / 3 * height))
    roi_bottom = int(floor(2 / 3 * height))

    # draw roi
    cv2.rectangle(clone, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0),
                  2)

    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    if not preprocessor.calibrated:
        if num_frames < 30:
            preprocessor.calibrate_background(frame)
        elif keypress == ord("c"):
            preprocessor.calibrate_object(frame)
    else:
        pp = preprocessor.preprocess(clone)
        cv2.imshow("preprocessed", pp)

    cv2.imshow("Video Feed", clone)

    num_frames += 1
# free up memory
camera.release()
cv2.destroyAllWindows()
