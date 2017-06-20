from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from daq.DatasetGenerator import gendata_sign
from daq.ImReader import get_paths_tm

dir_dataset = '../../resource/dataset/tm'
model_file = '../../resource/models/model.pkl'
# Reads dataset, trains models and stores them on file system
data, labels = gendata_sign(get_paths_tm(dir_dataset=dir_dataset), 2000)

scaler = StandardScaler()
extractor = PCA(0.9, svd_solver="full")
classifier = SVC()
pipe = Pipeline(steps=[('scaler', scaler), ('pca', extractor), ('clf', classifier)])
pipe.fit(data, labels.ravel())
joblib.dump(pipe, model_file)
