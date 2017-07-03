import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from preprocessing.representation.Descriptor import Descriptor
from preprocessing.representation.HistogramOfGradients import HistogramOfGradients


class BagOfHogs(Descriptor):
    def __init__(self, codebook_path=None, winsize=6, n_bins=16):
        self.n_bins = n_bins
        self.winsize = winsize
        if codebook_path is not None:
            self.codebook = joblib.load(codebook_path)

    def get_codebook(self, images, k_words=128, ):
        descriptors = np.vstack([HistogramOfGradients(self.winsize, self.n_bins).get_descr(img) for img in images])
        return KMeans(n_clusters=k_words).fit(descriptors).cluster_centers_

    @staticmethod
    def get_codebook_dist(hog, codebook, bandwidth=10000):
        dist = np.exp(-cdist(hog, codebook) / bandwidth)
        return np.max(dist, axis=0)

    def get_descr(self, img):
        return self.get_codebook_dist(HistogramOfGradients(self.winsize, self.n_bins).get_descr(img), self.codebook)
