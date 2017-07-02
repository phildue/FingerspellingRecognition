import random

import cv2

from datagen.FileProvider import FileProvider

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]

letter = str(random.choice(letters))
example_image_file = "../../../resource/dataset/tm/" + letter + str(
    random.choice(range(1, 40))) + ".tif"
# read image
img = FileProvider.read_img(example_image_file)

cv2.imshow('image', img)
img[img < 128] = 0
img[img >= 128] = 255
cv2.imshow("Likelihood", img)
cv2.waitKey(0)
