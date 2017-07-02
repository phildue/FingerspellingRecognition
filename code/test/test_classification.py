from numpy import mean, std
from sklearn.model_selection import cross_val_score

from classification.pipe import get_pipe
from daq.DatasetGenerator import load_data_sign, DatasetGenerator

data, labels = DatasetGenerator.load('../../resource/models/descriptors_pixel.pkl',
                                               '../../resource/models/labels.pkl')

model = get_pipe()
results = cross_val_score(model, data, labels.ravel(), cv=6)
print("Accuracy: " + str(mean(results)) + "(+/- " + str(std(results)))
