from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_transform(data, n_pc=0.9):
    # default 90% of variance is preserved
    scaler = StandardScaler()
    data_centered = scaler.fit_transform(data)
    pca = PCA(n_pc, svd_solver="full")
    return pca.fit_transform(data_centered)
