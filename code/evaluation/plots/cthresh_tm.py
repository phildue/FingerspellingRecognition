import random

import cv2

from datagen.FileProvider import FileProvider

example_image_file = "../../../resource/examples/d19.tif"
# read image
img = FileProvider.read_img(example_image_file)

cv2.imshow('image', img)
img[img < 128] = 0
img[img >= 128] = 255
cv2.imshow("Likelihood", img)
cv2.waitKey(0)
