import cv2
import numpy as np

from daq.hog import get_hog
from exceptions.exceptions import NoRoiFound, NoContoursFound


def extract_descriptors(imgs):
    return [extract_descriptor(img) for img in imgs]


def extract_descriptor(img):
    return get_hog(img).astype(dtype=np.uint8)


def preprocesss(imgs: [np.array], im_size=(100, 120), roi_size=(30, 30)) -> [np.array]:
    img_pp = []
    error_roi = 0
    for img in imgs:
        try:
            img_pp.append(preprocess(img, im_size, roi_size))
        except NoRoiFound:
            error_roi += 1
    if error_roi > 0:
        print("PreProcessing:: Could not find region of interest in " + str(error_roi) + " images")
    return img_pp


def preprocess(img: np.array, im_size=(100, 120), roi_size=(30, 30)) -> np.array:
    # find roi (hand), crop it, find edges
    img_resized = cv2.resize(img, im_size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    roi = get_roi(img_gray)
    roi = cv2.resize(roi, roi_size)
    # roi = cv2.Canny(roi, threshold1=50, threshold2=100)

    return roi


def get_longest_contours(image, n_longest=1):
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        raise NoContoursFound("get_longest_contours::No Contours found in image")
    longest_contours = []
    for i in range(0, n_longest):
        c_i = np.argmax([c.size for c in contours])
        longest_contours.append(contours[c_i])
    return longest_contours


def get_roi(img, min_roi_size=50):
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    img_edges = cv2.Canny(img_binary, threshold1=50, threshold2=100)
    _, contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if len(c) > 4]
    largest_contour = contours[np.argmax([cv2.contourArea(c) for c in contours])]

    x, y, w, h = cv2.boundingRect(largest_contour)
    #
    # cv2.imshow('roi', img_roi)
    # cv2.waitKey(10000)
    roi = img[y - 1:y + h + 1, x - 1:x + w + 1]
    if roi.size < min_roi_size:
        # img_roi = cv2.rectangle(img_edges, (x, y),(x+w,y-h), (0, 255, 0), 2, 1)
        # cv2.imshow("Error", img_roi)
        # cv2.waitKey(10000)
        raise NoRoiFound()

    return roi
