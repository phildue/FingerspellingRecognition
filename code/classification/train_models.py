from sklearn.externals import joblib

from classification.SignClassifier import SignClassifier
from daq.DatasetGenerator import gendata_sign
from daq.ImReader import get_paths_tm
from representation.FeatureExtraction import get_scaler, get_extractor

dir_dataset = '../../resource/dataset/tm'
scaler_file = '../../resource/models/scaler.pkl'
extractor_file = '../../resource/models/extractor.pkl'
classif_file = '../../resource/models/classif.pkl'
# Reads dataset, trains models and stores them on file system
data, labels = gendata_sign(get_paths_tm(dir_dataset=dir_dataset), 2000)
scaler = get_scaler(data)
extractor = get_extractor(data)
data = scaler.transform(data)
data = extractor.transform(data)
classif = SignClassifier()
classif.train(data, labels)
joblib.dump(scaler, scaler_file)
joblib.dump(extractor, scaler_file)
joblib.dump(classif.scikit_object, scaler_file)
