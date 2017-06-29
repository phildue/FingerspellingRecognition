import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from preprocessing.hog import get_hog


def gen_codebook(images, k_words=128):
    descriptors = np.vstack([get_hog(img, win_size=6, n_bins=16) for img in images])
    return KMeans(n_clusters=k_words).fit(descriptors).cluster_centers_


def get_codebook_dist(hog, codebook, bandwidth=10000):
    dist = np.exp(-cdist(hog, codebook) / bandwidth)
    return np.min(dist, axis=0)


def get_boh_descriptor(img, codebook_path):
    if get_boh_descriptor.codebook is None:
        get_boh_descriptor.codebook = joblib.load(codebook_path)

    return get_codebook_dist(get_hog(img, win_size=6, n_bins=16), get_boh_descriptor.codebook)


def get_boh_descriptors(imgs, codebook_path):
    return [get_boh_descriptor(img, codebook_path) for img in imgs]


get_boh_descriptor.codebook = None
