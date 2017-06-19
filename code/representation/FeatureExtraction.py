import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_transform(data, n_pc=0.9):
    # default 90% of variance is preserved
    scaler = StandardScaler()
    data_centered = scaler.fit_transform(data)
    pca = PCA(n_pc, svd_solver="full")
    return pca.fit_transform(data_centered)


def get_extractor(data, n_pc=0.9):
    pca = PCA(n_pc, svd_solver="full")
    pca.fit(data)
    return pca


def get_scaler(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler


def extract_features(descriptor, scaler_file: str, extractor_file):
    if extract_features.scaler is None:
        extract_features.scaler = pickle.load(scaler_file)
    if extract_features.extractor is None:
        extract_features.extractor = pickle.load(extractor_file)
    descriptor_scaled = extract_features.scaler.transform(descriptor)
    return extract_features.extractor.transform(descriptor_scaled)


extract_features.scaler = None
extract_features.extractor = None
