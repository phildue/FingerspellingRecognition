import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from preprocessing.representation.Descriptor import Descriptor
from preprocessing.representation.HistogramOfGradients import get_hog


class BagOfHogs(Descriptor):
    def __init__(self, codebook_path):
        self.codebook = joblib.load(codebook_path)

    @staticmethod
    def get_codebook(images, k_words=128):
        descriptors = np.vstack([get_hog(img, win_size=6, n_bins=16) for img in images])
        return KMeans(n_clusters=k_words).fit(descriptors).cluster_centers_

    @staticmethod
    def get_codebook_dist(hog, codebook, bandwidth=10000):
        dist = np.exp(-cdist(hog, codebook) / bandwidth)
        return np.min(dist, axis=0)

    def get_descr(self, img):
        return self.get_codebook_dist(get_hog(img, win_size=6, n_bins=16), self.codebook)
