import cv2
import maxflow as mf
import numpy as np

from preprocessing.segmentation.MarkovRandomField import MarkovRandomField
from preprocessing.segmentation.Segmenter import Segmenter


class MRFAsl(MarkovRandomField):
    def __init__(self, confidence_thresh=250):
        self.confidence_thresh = confidence_thresh

    def get_label(self, img):
        img_depth = img[1]
        img = img[0]
        img_segment = cv2.GaussianBlur(img, (11, 11), 2)

        threshold = img_depth[int(img_depth.shape[0] / 2), int(img_depth.shape[1] / 2)]

        img_gray = cv2.cvtColor(img_segment, cv2.COLOR_RGB2GRAY)

        background_label = img_gray.copy()
        background_label[img_depth != threshold] = 0
        background_label[img_gray > 235] = 0
        background_label[img_gray < 25] = 0

        # cv2.imshow("Without depth", background_label)
        area_foreground = background_label.copy()
        area_foreground[0:int(img_segment.shape[0] / 2 - 30), 0:int(img_segment.shape[1] / 2 - 30)] = 0
        area_foreground[int(img_segment.shape[0] / 2) + 30:img_segment.shape[0],
        int(img_segment.shape[1] / 2) + 30:img_segment.shape[0]] = 0

        pixels_fg = img_segment[area_foreground != 0].reshape(-1, 3)
        pixels_bg = img_segment[background_label == 0].reshape(-1, 3)

        likelihood_grid = self.get_weighted_sum(self.get_classifier_score(img_segment, pixels_fg, pixels_bg),
                                                self.get_background_score(threshold, img_depth))

        # cv2.imshow("Likelihood Foreground", (likelihood_grid[:, :, 1] * 255).astype(np.uint8))

        weight_x, weight_y = self.get_smooth_grid(img_segment)

        graph, nodeids = self.create_graph((img.shape[0], img.shape[1]), weight_x, weight_y, likelihood_grid[:, :, 1],
                                           likelihood_grid[:, :, 0])

        label_map = cv2.normalize(self.maxflow(graph, nodeids).astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        label_map = cv2.GaussianBlur(label_map, (3, 3), 2)

        kernel = np.ones((5, 5), np.uint8)

        label_map = cv2.morphologyEx(label_map, cv2.MORPH_CLOSE, kernel, iterations=5)

        _, label_map = cv2.threshold(label_map, self.confidence_thresh, 255, cv2.THRESH_BINARY)

        return label_map
