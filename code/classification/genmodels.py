from daq.imreader import get_paths_tm
from sklearn.externals import joblib

from classification.model import get_model
from daq.dataset.gendata import gendata_sign

dir_dataset = '../../resource/dataset/tm'
model_file = '../../resource/models/model.pkl'
# Reads dataset, trains models and stores them on file system
data, labels = gendata_sign(get_paths_tm(dir_dataset=dir_dataset), 2000)

pipe = get_model()
pipe.fit(data, labels.ravel())
joblib.dump(pipe, model_file)
