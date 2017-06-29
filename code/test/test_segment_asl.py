import cv2

from daq.fileaccess import read_image, read_image_depth
from preprocessing.segmentation import segment_asl

example_image_file = "../../resource/dataset/fingerspelling5/dataset5/E/h/color_7_0007.png"
img = read_image(example_image_file)

img_depth = read_image_depth(example_image_file)
cv2.imshow('image', img)

segmented = segment_asl(img, img_depth)

cv2.imshow('Segmented', segmented)

cv2.waitKey(0)
