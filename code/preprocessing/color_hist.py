import numpy as np
from numpy.linalg import norm


def colour_hist(img, n_bins=32):
    bin_range = 255 / n_bins
    hist = np.zeros(shape=(3, n_bins))
    for sample in img:
        for c in range(0, 3):
            hist[c][int(np.floor(sample[c] / bin_range))] += 1

    hist = hist.reshape(1, -1)
    hist /= norm(hist)
    return hist


def kl_divergence(hist1, hist2):
    eta = .0001 * np.ones(shape=hist1.shape)
    hist1_reg = hist1 + eta
    hist2_reg = hist2 + eta
    kl = np.multiply(hist1_reg, np.divide(hist1_reg, hist2_reg))
    kl[np.isinf(kl)] = 0
    kl[np.isnan(kl)] = 0
    return np.sum(kl)
