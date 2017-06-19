import cv2
import numpy as np

from exceptions.exceptions import NoRoiFound, NoContoursFound


def extract_descriptors(imgs):
    descriptors = []
    error_roi = 0
    for img in imgs:
        try:
            descriptors.append(extract_descriptor(img))
        except NoRoiFound:
            error_roi += 1
    if error_roi > 0:
        print("ExtractDescriptors:: Could not find region of interest in " + str(error_roi) + " images")
    return descriptors


def extract_descriptor(img):
    img = prefilter(img)
    return img.reshape(1, img.size)


def prefilter(img, im_size=(100, 120), roi_size=(30, 30)):
    # find roi (hand), crop it, find edges
    img_resized = cv2.resize(img, im_size)

    roi = get_roi(img_resized)
    roi = cv2.resize(roi, roi_size)
    roi = cv2.Canny(roi, threshold1=50, threshold2=100)

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
