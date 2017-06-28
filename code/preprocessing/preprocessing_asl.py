import cv2
import numpy as np

from exceptions.exceptions import NoRoiFound
from preprocessing.hog import get_hog
from preprocessing.segmentation import segment_asl
from preprocessing.segmentation_tm import get_roi


def extract_descriptors(imgs):
    return [extract_descriptor(img) for img in imgs]


def extract_descriptor(img):
    return get_hog(img).astype(dtype=np.uint8)


def preprocesss(imgs: [(np.array, np.array)], im_size=(100, 120), roi_size=(30, 30)) -> [np.array]:
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


def preprocess(img: (np.array, np.array), im_size=(100, 120), roi_size=(30, 30)) -> np.array:
    # find roi (hand), crop it, find edges
    img_segment = segment_asl(img[0], img[1])
    img_segment = cv2.cvtColor(img_segment, cv2.COLOR_RGB2GRAY)
    return cv2.resize(img_segment, roi_size)
