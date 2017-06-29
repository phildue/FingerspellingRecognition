from sklearn.externals import joblib

from daq.fileaccess import get_paths_asl
from daq.gendata import gendata_sign

n_data = 2500
data, labels = gendata_sign(get_paths_asl("../../resource/dataset/fingerspelling5/dataset5/"), n_data)

joblib.dump(data, '../../resource/models/descriptors.pkl')
joblib.dump(labels, '../../resource/models/labels.pkl')
