import cv2
import numpy as np
from numpy.linalg import linalg

from exceptions.exceptions import NoRoiFound


def extract_descriptors(imgs):
    descriptors = []
    error = 0
    for img in imgs:
        try:
            descriptors.append(extract_descriptor(img))
        except NoRoiFound:
            error += 1

    if error > 0:
        print("Could not find region of interest in " + str(error) + " images")

    return descriptors


def extract_descriptor(img):
    img = prefilter(img)
    contour = get_longest_contours(img)[0]
    centroid = get_centroid(contour)
    return get_centroid_distances(contour, centroid)


def get_centroid(contour):
    moms = cv2.moments(contour)
    return np.array([moms['m01'] / moms['m00'], moms['m10'] / moms['m00']])


def get_centroid_distances(contour, centroid):
    distances = []
    for p in contour:
        distances.append(linalg.norm(centroid - p))

    return distances


def prefilter(img, im_size=(100, 120), roi_size=(30, 30)):
    img_resized = cv2.resize(img, im_size)

    roi = get_roi(img_resized)
    roi = cv2.resize(roi, roi_size)
    roi = cv2.Canny(roi, threshold1=50, threshold2=100)
    # _, roi = cv2.threshold(roi, 1, 255, cv2.THRESH_BINARY)
    # contour_img = np.zeros(roi.shape, np.uint8)

    return roi


def get_longest_contours(image, n_longest=1):
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    longest_contours = []
    for i in range(0, n_longest):
        c_i = np.argmax([c.size for c in contours])
        longest_contours.append(contours[c_i])
    return longest_contours


def get_roi(img, min_roi_size=100):
    _, img_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    img_edges = cv2.Canny(img_binary, threshold1=50, threshold2=100)
    _, contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
