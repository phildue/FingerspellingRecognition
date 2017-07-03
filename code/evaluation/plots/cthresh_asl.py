import cv2
import numpy as np

from datagen.FileProviderAsl import FileProviderAsl


def colourbased_skin_segmentation(image, lower_thresh=np.array([0, 80, 80], dtype="uint8"),
                                  upper_thresh=np.array([20, 240, 240], dtype="uint8")):
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(converted, lower_thresh, upper_thresh)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 2)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    _, likelihood = cv2.threshold(skin_mask, 10, 255, cv2.NORM_MINMAX)
    return likelihood


example_image_file = "../../../resource/examples/color_0_0028.png"
# read image
img, _ = FileProviderAsl.read_img(example_image_file)

cv2.imshow('image', img)
img = colourbased_skin_segmentation(img)
cv2.imshow("Likelihood", img.astype(np.uint8))
cv2.waitKey(0)
