from sklearn.externals import joblib

from classification.pipe import get_pipe
from daq.gendata import gendata_sign, load_data_sign

model_file = '../../resource/models/model_hog.pkl'
# Reads dataset, trains models and stores them on file system
data, labels = load_data_sign('../../resource/models/descriptors_hog.pkl', '../../resource/models/labels.pkl')


pipe = get_pipe()
pipe.fit(data, labels.ravel())
joblib.dump(pipe, model_file)
