from daq.datageneration.ImReader import get_paths_tm
from numpy import mean

from classification.model import get_model
from daq.gendata import gendata_sign
from evaluation.methods import crossval

n_data = 100
data, labels = gendata_sign(get_paths_tm(dir_dataset='../../resource/dataset/tm'
                                         ), n_data)

model = get_model()
print("Accuracy: " + str(mean(crossval(model, data, labels,folds=6))))



