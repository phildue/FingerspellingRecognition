import cv2
import numpy as np
from numpy.linalg import norm
from numpy.ma import floor

from daq.fileaccess import get_paths_asl, read_image
from preprocessing.segmentation.ColourHistogram import ColourHistogram

paths_dict = get_paths_asl(sets=["E"], alphabet=["a"])
winsize = 3
samples = []
for paths in paths_dict.values():
    for path in paths:
        img = read_image(path)
        img_filtered = cv2.pyrMeanShiftFiltering(img, 5, 50)

        y = int(floor(img.shape[0] / 2) - floor(winsize / 2))
        x = int(floor(img.shape[1] / 2) - floor(winsize / 2))
        # sample = img.copy()
        # cv2.rectangle(sample, (x, y), (x+winsize, y+winsize), (0, 255, 0))
        # cv2.imshow("Box", sample)

        window = img[y:y + winsize, x:x + winsize]
        samples.append(window.reshape(-1, 3))

n_bins = 32
bin_range = 255 / n_bins
hist = np.zeros(shape=(1, 3 * n_bins))
for sample in samples:
    hist += ColourHistogram.colour_hist(sample)

hist = hist.reshape(1, -1)
hist /= norm(hist)
print(str(hist))
np.save("../../../resource/models/skinhist_asl.npy", hist)
