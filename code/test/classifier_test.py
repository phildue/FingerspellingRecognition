from numpy import mean, std
from sklearn.model_selection import cross_val_score

from classification.pipe import get_pipe
from daq.dataset.fileaccess import get_paths_tm
from daq.dataset.gendata import gendata_sign

n_data = 100
data, labels = gendata_sign(get_paths_tm(dir_dataset='../../resource/dataset/tm'
                                         ), n_data)

model = get_pipe()
results = cross_val_score(model, data, labels.ravel(), cv=6)
print("Accuracy: " + str(mean(results)) + "(+/- " + str(std(results)))
