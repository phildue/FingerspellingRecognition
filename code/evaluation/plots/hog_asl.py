import cv2

from datagen.FileProviderAsl import FileProviderAsl
from preprocessing.PreProcessorAsl import PreProcessorAsl
from preprocessing.representation.HistogramOfGradients import HistogramOfGradients
from preprocessing.segmentation.MRFAsl import MRFAsl

example_image_file = "../../../resource/dataset/fingerspelling5/dataset5/A/a/color_0_0027.png"
# read image
img = FileProviderAsl.read_img(example_image_file)

cv2.imshow('image', img[0])

pp = PreProcessorAsl(img_size=(120, 120), segmenter=MRFAsl(),
                     descriptor=HistogramOfGradients(window_size=6, n_bins=8))

img = pp.preprocess(img)
descr = pp.get_descr(img)
cv2.imshow("Histogram of Gradients", descr.reshape(100, -1))
print(str(descr))
cv2.waitKey(30000)
cv2.destroyAllWindows()
exit(0)
