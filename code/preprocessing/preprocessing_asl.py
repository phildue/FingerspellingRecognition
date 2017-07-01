import cv2
import numpy as np

from exceptions.exceptions import NoRoiFound
from preprocessing.representation.HistogramOfGradients import get_hog
from preprocessing.segmentation import segment_asl


def extract_descriptors(imgs):
    return [extract_descriptor(img) for img in imgs]


def extract_descriptor(img):
    return get_hog(img, win_size=6, n_bins=16).reshape(1, -1)


def preprocesss(imgs: [(np.array, np.array)], roi_size=(60, 60)) -> [np.array]:
    img_pp = []
    error_roi = 0
    for img in imgs:
        try:
            img_pp.append(preprocess(img, roi_size))
        except NoRoiFound:
            error_roi += 1
    if error_roi > 0:
        print("PreProcessing:: Could not find region of interest in " + str(error_roi) + " images")
    return img_pp


def preprocess(img: (np.array, np.array), roi_size=(60, 60)) -> np.array:
    # find roi (hand), crop it, find edges
    img_segment = segment_asl(img[0], img[1])
    img_segment = cv2.cvtColor(img_segment, cv2.COLOR_RGB2GRAY)
    img_segment = cv2.resize(img_segment, roi_size)
    return cv2.equalizeHist(img_segment)
