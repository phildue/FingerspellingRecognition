import cv2
import numpy as np


def pre_processing(img, im_size=(100, 120)):
    img_resized = cv2.resize(img, (im_size[0], im_size[1]))
    cv2.imshow('image', img_resized)
    cv2.waitKey(10000)
    img_preprocessed = cv2.blur(img_resized, (7, 7))
    cv2.imshow('image', img_preprocessed)
    cv2.waitKey(10000)
    img_preprocessed = cv2.pyrMeanShiftFiltering(img_preprocessed, 30, 30)
    cv2.imshow('image', img_preprocessed)
    cv2.waitKey(10000)
    img_preprocessed = filter_skin(img_preprocessed)
    cv2.imshow('image', img_preprocessed)
    cv2.waitKey(10000)
    img_preprocessed = cv2.cvtColor(img_preprocessed, cv2.COLOR_RGB2GRAY)
    cv2.imshow('image', img_preprocessed)
    cv2.waitKey(10000)
    img_preprocessed = cv2.Canny(img_preprocessed, threshold1=50, threshold2=100)
    _, img_preprocessed = cv2.threshold(img_preprocessed, 10, 255, cv2.THRESH_BINARY)
    cv2.imshow('image', img_preprocessed)
    cv2.waitKey(10000)
    _, contours, hierarchy = cv2.findContours(img_preprocessed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lens = []
    for c in contours: lens.append(len(c))
    contours = [c for c in contours if len(c) >= 50]
    img_preprocessed = cv2.drawContours(img_resized, contours, -1, (0, 255, 0), 2)
    cv2.imshow('image', img_preprocessed)
    cv2.waitKey(10000)
    # TODO findContours/remove blobs/ calculate distance to centroid for each point on the contour

    return img_preprocessed


def filter_skin(image, lower_thresh=np.array([0, 40, 40], dtype="uint8"),
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
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 2)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    return skin


def equalize_intensity(image):
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    channels = cv2.split(img_ycrcb)

    img_equalized = cv2.equalizeHist(channels[0], channels[0])

    cv2.merge(img_equalized, img_ycrcb)

    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2RGB)
