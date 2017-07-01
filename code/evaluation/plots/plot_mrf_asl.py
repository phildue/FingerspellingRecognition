import random

import cv2
import numpy as np

from daq.fileaccess import read_image, read_image_asl
from preprocessing.mrf import MarkovRandomField
from preprocessing.segmentation import get_classifier_score, get_background_score, get_weighted_sum, get_smooth_grid, \
    extract_label


def colourbased_skin_segmentation(image, lower_thresh=np.array([0, 80, 80], dtype="uint8"),
                                  upper_thresh=np.array([20, 240, 240], dtype="uint8")):
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(converted, lower_thresh, upper_thresh)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 2)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    _, likelihood = cv2.threshold(skin_mask, 10, 255, cv2.NORM_MINMAX)
    return likelihood


example_image_file = "../../../resource/dataset/fingerspelling5/dataset5/A/a/color_0_0028.png"
# read image
img, img_depth = read_image_asl(example_image_file)

cv2.imshow('image', img)

img_segment = cv2.GaussianBlur(img, (11, 11), 2)

threshold = img_depth[int(img_depth.shape[0] / 2), int(img_depth.shape[1] / 2)]

img_gray = cv2.cvtColor(img_segment, cv2.COLOR_RGB2GRAY)

background_label = img_gray.copy()
background_label[img_depth != threshold] = 0
background_label[img_gray > 235] = 0
background_label[img_gray < 25] = 0

area_foreground = background_label.copy()
area_foreground[0:int(img_segment.shape[0] / 2 - 30), 0:int(img_segment.shape[1] / 2 - 30)] = 0
area_foreground[int(img_segment.shape[0] / 2) + 30:img_segment.shape[0],
int(img_segment.shape[1] / 2) + 30:img_segment.shape[0]] = 0

pixels_fg = img_segment[area_foreground != 0].reshape(-1, 3)
pixels_bg = img_segment[background_label == 0].reshape(-1, 3)

classifier_score = get_classifier_score(img_segment, pixels_fg, pixels_bg)
background_score = get_background_score(threshold, img_depth)

combined_likelihood = get_weighted_sum(classifier_score, background_score)

weight_x, weight_y = get_smooth_grid(img_segment)

labels = MarkovRandomField((img_segment.shape[0], img_segment.shape[1]), weight_x, weight_y,
                           combined_likelihood[:, :, 1],
                           combined_likelihood[:, :, 0]).maxflow()
segmented_image = extract_label(img, labels)

classifier_score = cv2.normalize(classifier_score, None, 0.0, 255.0, cv2.NORM_MINMAX)
background_score = cv2.normalize(background_score, None, 0.0, 255.0, cv2.NORM_MINMAX)
combined_likelihood = cv2.normalize(combined_likelihood, None, 0.0, 255.0, cv2.NORM_MINMAX)
weight_x *= -1
weight_y *= -1
cv2.imshow("Knn Likelihood", classifier_score.astype(np.uint8)[:, :, 1])
cv2.imshow("Depth-Segmentation Likelihood", background_score.astype(np.uint8)[:, :, 1])
cv2.imshow("Combined Likelihood", combined_likelihood.astype(np.uint8)[:, :, 1])
cv2.imshow("Ix", weight_x.astype(np.uint8))
cv2.imshow("Iy", weight_y.astype(np.uint8))
cv2.imshow("Segmented image", segmented_image)
cv2.waitKey(0)
