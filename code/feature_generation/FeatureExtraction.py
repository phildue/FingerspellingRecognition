from sklearn.decomposition import PCA


def pca_transform(data, n_pc=0.9):
    # default 90% of variance is preserved
    pca = PCA(n_pc, svd_solver="full")
    return pca.fit_transform(data)
