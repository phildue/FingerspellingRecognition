from sklearn.decomposition import PCA


def pca_transform(data):
    pca = PCA(n_components=0.9, svd_solver="full")
    return pca.fit_transform(data)
