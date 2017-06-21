import cv2
import numpy as np


def get_hog_window(window, n_bins=8):
    dx = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=5)
    bin_range = 2 * np.pi / n_bins
    bins = np.zeros(shape=(1, n_bins))
    for y in range(0, window.shape[0] - 1):
        for x in range(0, window.shape[1] - 1):
            angle, _, _ = np.arctan2(dx[x, y], dy[x, y])
            if angle < 0:
                angle += 2 * np.pi
            bin = int(np.floor(angle / bin_range))
            bins[0, bin] += np.sqrt(dx[y, x][0] ** 2 + dy[y, x][0] ** 2)
    return bins


def get_hog(img, window_size=(3, 3), n_bins=8, padding=1):
    padded_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_WRAP)
    hog = np.zeros(shape=(10, n_bins))
    w_i = 0
    for y in range(1, padded_img.shape[0] - 1, window_size[0]):
        for x in range(1, padded_img.shape[1] - 1, window_size[1]):
            hog[w_i, :] = get_hog_window(
                padded_img[y - 1:y + 2, x - 1:x + 2], n_bins)

    return hog.reshape(1, 10 * 8)
