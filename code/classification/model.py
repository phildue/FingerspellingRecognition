from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_model():
    scaler = StandardScaler()
    extractor = PCA(0.9, svd_solver="full")
    classifier = SVC()
    pipe = Pipeline(steps=[('scaler', scaler), ('pca', extractor), ('clf', classifier)])
    return pipe
