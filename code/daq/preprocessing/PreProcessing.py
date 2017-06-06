import cv2
import numpy as np


def pre_processing(img, im_size=(100, 120)):
    img_preprocessed = cv2.resize(img, (im_size[0], im_size[1]))
    img_preprocessed = cv2.pyrMeanShiftFiltering(img_preprocessed, 25, 25)
    img_preprocessed = filter_skin(img_preprocessed)
    img_preprocessed = cv2.cvtColor(img_preprocessed, cv2.COLOR_RGB2GRAY)
    img_preprocessed = cv2.blur(img_preprocessed, (7, 7))
    img_preprocessed = cv2.Canny(img_preprocessed, threshold1=50, threshold2=100)
    _, img_preprocessed = cv2.threshold(img_preprocessed, 50, 255, cv2.THRESH_BINARY)
    # TODO findContours/remove blobs/ calculate distance to centroid for each point on the contour
    return img_preprocessed


def filter_skin(image, lower_thresh=np.array([0, 80, 80], dtype="uint8"),
                upper_thresh=np.array([20, 170, 170], dtype="uint8")):
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
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 2)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    return skin
