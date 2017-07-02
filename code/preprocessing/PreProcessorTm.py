import cv2
import numpy as np

from exceptions.exceptions import NoRoiFound, NoContoursFound
from preprocessing.PreProcessor import PreProcessor
from preprocessing.representation.Descriptor import Descriptor
from preprocessing.representation.HistogramOfGradients import HistogramOfGradients
from preprocessing.segmentation.GrayThresh import GrayThresh


class PreProcessorTm(PreProcessor):
    def __init__(self, descriptor: Descriptor = HistogramOfGradients(), img_size=(30, 30),
                 segmenter=GrayThresh()):
        PreProcessor.__init__(self)
        self.segmenter = segmenter
        self.descriptor = descriptor
        self.roi_size = img_size

    def get_descr(self, img):
        self.descriptor.get_descr(img)

    def preprocess(self, img):
        # find roi (hand), crop it, find edges

        labels = self.segmenter.get_label(img)
        labels = self.crop_roi(labels)
        labels = cv2.resize(labels, self.roi_size)

        return labels

    def preprocess_all(self, imgs: [np.array]) -> [np.array]:
        img_pp = []
        error_roi = 0
        for img in imgs:
            try:
                img_pp.append(self.preprocess(img))
            except NoRoiFound:
                error_roi += 1
        if error_roi > 0:
            print("PreProcessing:: Could not find region of interest in " + str(error_roi) + " images")
        return img_pp

    @staticmethod
    def get_longest_contours(image, n_longest=1):
        _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            raise NoContoursFound("get_longest_contours::No Contours found in image")
        longest_contours = []
        for i in range(0, n_longest):
            c_i = np.argmax([c.size for c in contours])
            longest_contours.append(contours[c_i])
        return longest_contours

    @staticmethod
    def crop_roi(img, min_roi_size=50):
        img_edges = cv2.Canny(img, threshold1=50, threshold2=100)
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
