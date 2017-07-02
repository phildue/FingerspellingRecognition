import random

import cv2

from datagen.FileProvider import FileProvider
from preprocessing.PreProcessorTm import PreProcessorTm

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]

letter = str(random.choice(letters))
example_image_file = "../../resource/dataset/tm/" + letter + str(
    random.choice(range(1, 40))) + ".tif"
# read image
img = FileProvider.read_img(example_image_file)
pp = PreProcessorTm(img_size=(60, 60))

cv2.imshow('image', img)
img = pp.preprocess(img)
cv2.imshow("after prefiltering", img)
cv2.waitKey(0)

descriptor = pp.get_descr(img)
print("Descriptor: \n" + str(descriptor))
# print("dim: \n" + str(descriptor.shape[1]))
cv2.waitKey(10000)
cv2.destroyAllWindows()
exit(0)
