import random

import cv2

from daq.fileaccess import read_image, read_image_asl
from preprocessing.preprocessing_asl import preprocess, extract_descriptor

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]

example_image_file = "../../resource/dataset/fingerspelling5/dataset5/A/a/color_0_0027.png"
# read image
img = read_image_asl(example_image_file)
cv2.imshow('depth', cv2.normalize(img[1], None, 0, 255, cv2.NORM_MINMAX))

cv2.imshow('image', img[0])

img = preprocess(img, roi_size=(120, 120))
cv2.imshow("after prefiltering", img)
cv2.waitKey(0)

descriptor = extract_descriptor(img)
print("Descriptor: \n" + str(descriptor))
# print("dim: \n" + str(descriptor.shape[1]))
cv2.waitKey(10000)
cv2.destroyAllWindows()
exit(0)
