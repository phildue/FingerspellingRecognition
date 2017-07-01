from sklearn.externals import joblib

from classification.pipe import get_pipe
from daq.gendata import gendata_sign, load_data_sign

dir_dataset = '../../resource/dataset/tm'
model_file = '../../resource/models/model_hog_asl.pkl'
# Reads dataset, trains models and stores them on file system
data, labels = load_data_sign('../../resource/models/descriptors_pixel.pkl', '../../resource/models/labels.pkl', 2500)


pipe = get_pipe()
pipe.fit(data, labels.ravel())
joblib.dump(pipe, model_file)
