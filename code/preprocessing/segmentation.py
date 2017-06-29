from numpy import random

import cv2
import numpy as np
from numpy.linalg import linalg
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier

from preprocessing.mrf import MarkovRandomField


def get_smooth_grid(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.normalize(img_gray.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX)
    img_gray = cv2.GaussianBlur(img_gray, (11, 11), 5)
    dx = cv2.Sobel(img_gray, -1, 0, 1, ksize=3)
    dx = np.abs(dx)
    dx[dx < 20] = 0
    dy = cv2.Sobel(img_gray, -1, 1, 0, ksize=3)
    dy = np.abs(dy)
    dy[dy < 20] = 0
    wx = -cv2.normalize(dx.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX)
    wy = -cv2.normalize(dy.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX)
    return wx, wy


def get_background_score(img, background, bandwidth=10):
    background_score_grid = np.zeros(shape=(background.shape[0], background.shape[1], 2))
    background_score_grid[:, :, 0] = np.exp(-linalg.norm(img - background) / bandwidth)
    background_score_grid[:, :, 1] = 1 - background_score_grid[:, :, 0]

    return background_score_grid


def get_classifier_score(img, pixels_fg, pixels_bg):
    pixels_fg = pixels_fg[random.randint(pixels_fg.shape[0], size=200), :]
    pixels_bg = pixels_bg[random.randint(pixels_bg.shape[0], size=1000), :]

    data = np.vstack([pixels_bg, pixels_fg])
    labels = np.vstack([np.zeros(shape=(pixels_bg.shape[0], 1)), np.ones(shape=(pixels_fg.shape[0], 1))])

    knn = KNeighborsClassifier(3).fit(data, labels.ravel())

    return knn.predict_proba(img.reshape(-1, 3)).reshape(img.shape[0], img.shape[1], 2)


def get_weighted_sum(classifier_score_grid, background_score_grid, w_classifier=0.8):
    w_background = 1 - w_classifier
    score_grid = w_classifier * classifier_score_grid + w_background * background_score_grid

    return cv2.normalize(score_grid.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX)


def mrf_segmentation(img, img_depth):
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

    likelihood_grid = get_weighted_sum(get_classifier_score(img_segment, pixels_fg, pixels_bg),
                                       get_background_score(threshold, img_depth))

    # cv2.imshow("Likelihood Foreground", (likelihood_grid[:, :, 1] * 255).astype(np.uint8))

    weight_x, weight_y = get_smooth_grid(img_segment)

    return MarkovRandomField((img_segment.shape[0], img_segment.shape[1]), weight_x, weight_y, likelihood_grid[:, :, 1],
                             likelihood_grid[:, :, 0]).maxflow()


def extract(img, label_map):
    label_map = cv2.normalize(label_map.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow("Labels", label_map)
    label_map = cv2.GaussianBlur(label_map, (3, 3), 2)
    kernel = np.ones((1, 1), np.uint8)
    label_map = cv2.erode(label_map, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)

    label_map = cv2.morphologyEx(label_map, cv2.MORPH_CLOSE, kernel, iterations=2)

    _, label_map = cv2.threshold(label_map, 250, 255, cv2.THRESH_BINARY)

    img_extracted = img.copy()
    img_extracted[label_map == 0] = 0
    return img_extracted


def segment_asl(img, img_depth):
    img_labeled = mrf_segmentation(img, img_depth)

    return extract(img, img_labeled)
