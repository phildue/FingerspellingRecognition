import cv2

from daq.fileaccess import read_image_asl
from preprocessing.preprocessing_asl import preprocess
from preprocessing.representation.HistogramOfGradients import get_hog

example_image_file = "../../../resource/dataset/fingerspelling5/dataset5/A/a/color_0_0027.png"
# read image
img = read_image_asl(example_image_file)

cv2.imshow('image', img[0])

img = preprocess(img, roi_size=(120, 120))
img = get_hog(img, win_size=6, n_bins=16)
cv2.imshow("Histogram of Gradients", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit(0)
