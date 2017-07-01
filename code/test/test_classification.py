from numpy import mean, std
from sklearn.model_selection import cross_val_score

from classification.pipe import get_pipe
from daq.gendata import load_data_sign

n_data = 2500
data, labels = load_data_sign('../../resource/models/descriptors.pkl', '../../resource/models/labels.pkl', n_data)

model = get_pipe()
results = cross_val_score(model, data, labels.ravel(), cv=6)
print("Accuracy: " + str(mean(results)) + "(+/- " + str(std(results)))
