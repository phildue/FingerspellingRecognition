import cv2
from numpy import shape

from daq.preprocessing.SkinSegmentor import filter_skin


def pre_processing(img, im_size = (100, 120)):
    img_preprocessed = cv2.resize(img, (im_size[0], im_size[1]))
    img_preprocessed = filter_skin(img_preprocessed)
    img_preprocessed = cv2.cvtColor(img_preprocessed, cv2.COLOR_RGB2GRAY)
    return img_preprocessed
