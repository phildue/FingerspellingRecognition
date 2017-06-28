from sklearn.externals import joblib

from classification.pipe import get_pipe
from daq.dataset.fileaccess import get_paths_tm
from daq.dataset.gendata import gendata_sign

dir_dataset = '../../resource/dataset/tm'
model_file = '../../resource/models/model.pkl'
# Reads dataset, trains models and stores them on file system
data, labels = gendata_sign(get_paths_tm(dir_dataset=dir_dataset), 40,
                            letters=["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q",
                                     "r", "s", "t",
                                     "u",
                                     "v", "w", "x", "y"])

pipe = get_pipe()
pipe.fit(data, labels.ravel())
joblib.dump(pipe, model_file)
