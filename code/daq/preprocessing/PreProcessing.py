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


def extract_descriptor(img, n_points=50):
    img = prefilter(img)
    contour = get_longest_contours(img)[0]
    centroid = get_centroid(contour)
    return get_centroid_distances(contour, centroid, n_points)


def get_centroid(contour):
    moms = cv2.moments(contour)
    return np.array([moms['m01'] / moms['m00'], moms['m10'] / moms['m00']])


def get_centroid_distances(contour, centroid, n_points):
    # calculate the distance of every point to the centroid
    #  this results in rotation invariant features
    points = get_equally_distr_points(contour, n_points)

    distances = []
    for p in points:
        distances.append(linalg.norm(p - centroid))

    return distances


def get_equally_distr_points(contour, n):
    # get points along the found contour in equally distributed
    # distances. This is required to have the same amount of
    # features for every shape
    total_dist = float(0)
    for n_contour in range(1, len(contour)):
        total_dist += np.linalg.norm(contour[n_contour] - contour[n_contour - 1])

    delta_dist = total_dist / (n-1)
    points = [contour[0]]
    current_dist = float(0)
    n_contour = 0
    current_p = contour[0].astype(float)
    while current_dist < total_dist:
        last_p = current_p.copy()

        if linalg.norm(contour[n_contour] - current_p) + current_dist < delta_dist * len(points):
            current_p = contour[n_contour]
            n_contour += 1
        else:
            vnext = contour[n_contour] - current_p
            step = (delta_dist * len(points) - current_dist) / linalg.norm(vnext)
            current_p = np.add(current_p, step * vnext)
            points.append(current_p)

        current_dist += linalg.norm(current_p - last_p)
    return points


def prefilter(img, im_size=(100, 120), roi_size=(30, 30)):
    # find roi (hand), crop it, find edges
    img_resized = cv2.resize(img, im_size)

    roi = get_roi(img_resized)
    roi = cv2.resize(roi, roi_size)
    roi = cv2.Canny(roi, threshold1=50, threshold2=100)

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
