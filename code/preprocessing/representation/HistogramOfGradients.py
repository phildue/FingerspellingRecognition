import cv2
import numpy as np

from preprocessing.representation.Descriptor import Descriptor


class HistogramOfGradients(Descriptor):
    def __init__(self, window_size=6, n_bins=8):
        self.window_size = window_size
        self.n_bins = n_bins

    def get_descr(self, img: np.array):
        return self.get_hog(img, self.window_size, self.n_bins)
        pass

    @staticmethod
    def get_hog_window(window, n_bins=8):
        dx = cv2.Sobel(window, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(window, cv2.CV_32F, 0, 1, ksize=3)

        bin_range = 2 * np.pi / n_bins
        bins = np.zeros(shape=(1, n_bins))
        magn = cv2.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dx, dy)
        for y in range(0, window.shape[0]):
            for x in range(0, window.shape[1]):
                angle_shift = angle[y, x]
                if angle_shift < 0:
                    angle_shift += 2 * np.pi
                bin = int(np.floor(angle_shift / bin_range))
                bins[0, bin] += magn[y, x]
        return bins

    @staticmethod
    def get_hog(img, win_size=3, n_bins=8):
        step = int(np.floor(win_size / 2))
        padded_img = cv2.copyMakeBorder(img, step, step, step, step, cv2.BORDER_WRAP)
        windows = int(np.floor(img.shape[0] / win_size) * np.floor(img.shape[1] / win_size))
        hog = np.zeros(shape=(windows, n_bins))
        w_i = 0
        for y in range(step, img.shape[0], win_size):
            for x in range(step, img.shape[1], win_size):
                hog[w_i, :] = HistogramOfGradients.get_hog_window(
                    padded_img[y - step:y + step + 1, x - step:x + step + 1], n_bins)
                w_i += 1

        return hog
