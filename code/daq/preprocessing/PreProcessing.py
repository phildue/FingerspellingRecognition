import cv2
import numpy as np

from exceptions.exceptions import NoRoiFound


def preprocess(imgs, roi_size=(30, 30)):
    imgs_preprocessed = []
    error = 0
    for i, img in enumerate(imgs):
        try:
            imgs_preprocessed.append(preprocess_img(img, roi_size))
        except NoRoiFound:
            error += 1

    if error > 0:
        print("Could not find region of interest in " + str(error) + " images")

    return imgs_preprocessed


def preprocess_img(img, im_size=(100, 120), roi_size=(30, 30)):
    img_resized = cv2.resize(img, im_size)

    roi = get_roi(img_resized)
    roi = cv2.resize(roi, roi_size)
    roi = cv2.Canny(roi, threshold1=50, threshold2=100)

    return roi


def get_roi(img, min_roi_size=100):
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    img_edges = cv2.Canny(img_binary, threshold1=50, threshold2=100)
    _, contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = contours[np.argmax([cv2.contourArea(c) for c in contours])]

    x, y, w, h = cv2.boundingRect(largest_contour)
    # img_roi = cv2.rectangle(img_binary, x, y, (0, 0, 255), 2, 1)
    # cv2.imshow('roi', img_roi)
    # cv2.waitKey(10000)
    roi = img[y - 1:y + h + 1, x - 1:x + w + 1]
    if roi.size < min_roi_size:
        # cv2.imshow("Error", roi)
        # cv2.waitKey(10000)
        raise NoRoiFound()

    return roi
