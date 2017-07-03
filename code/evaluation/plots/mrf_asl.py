import cv2
import numpy as np

from datagen.FileProviderAsl import FileProviderAsl
from preprocessing.segmentation.MRFAsl import MRFAsl

example_image_file = "../../../resource/examples/color_0_0028.png"
# read image
img, img_depth = FileProviderAsl.read_img(example_image_file)

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
segmenter = MRFAsl()
classifier_score = segmenter.get_classifier_score(img_segment, pixels_fg, pixels_bg)
background_score = segmenter.get_background_score(threshold, img_depth)

combined_likelihood = segmenter.get_weighted_sum(classifier_score, background_score)

weight_x, weight_y = segmenter.get_smooth_grid(img_segment)

graph, nodes = segmenter.create_graph((img_segment.shape[0], img_segment.shape[1]), weight_x, weight_y,
                                      combined_likelihood[:, :, 1],
                                      combined_likelihood[:, :, 0])

label_map = cv2.normalize(segmenter.maxflow(graph, nodes).astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
label_map = cv2.GaussianBlur(label_map, (3, 3), 2)

kernel = np.ones((5, 5), np.uint8)

label_map = cv2.morphologyEx(label_map, cv2.MORPH_CLOSE, kernel, iterations=5)
_, label_map = cv2.threshold(label_map, 250, 255, cv2.THRESH_BINARY)
img_extracted = img.copy()
img_extracted[label_map == 0] = 0

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
cv2.imshow("Segmented image", img_extracted)
cv2.waitKey(0)
