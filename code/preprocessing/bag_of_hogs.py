import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from preprocessing.hog import get_hog
from preprocessing.preprocessing_asl import preprocess


def gen_codebook(images, k_words=128):
    return KMeans(n_clusters=k_words).fit(
        np.vstack([get_hog(preprocess(img, roi_size=(60, 60)), win_size=6, n_bins=16) for img in images])
    ).cluster_centers_


def get_codebook_dist(hog, codebook, bandwidth=100000):
    dist = np.exp(-cdist(hog, codebook) / bandwidth)
    return np.min(dist, axis=0)
