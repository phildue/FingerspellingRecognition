import cv2

from daq.FileProviderAsl import FileProviderAsl
from preprocessing.segmentation.MRFAsl import MRFAsl

example_image_file = "../../resource/dataset/fingerspelling5/dataset5/A/a/color_0_0026.png"
img = FileProviderAsl.read_img(example_image_file)

img_depth = img[1]
cv2.imshow('image', img)

segmented = MRFAsl().get_label(img)

cv2.imshow('Segmented', segmented)

cv2.waitKey(0)
