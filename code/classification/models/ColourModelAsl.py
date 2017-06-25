import numpy as np
from numpy.linalg import norm, inv
from scipy.stats import entropy

from preprocessing.colourhist import colour_hist, kl_divergence


class ColourModelAsl:
    def __init__(self, filepath="../../resource/dataset/skin/skinhist_asl"):
        self.skinhist = np.load(filepath)

    def get_likelihood(self, img, sigma=500):
        return np.exp(kl_divergence(self.skinhist, colour_hist(img)) / sigma)
