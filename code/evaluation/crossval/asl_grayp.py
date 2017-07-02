from numpy import mean, std
from sklearn.model_selection import cross_val_score

from classification.pipe import get_pipe
from datagen.fileaccess import get_paths_asl
from datagen.DatasetGenerator import load_data_sign, gendata_sign

n_data = 2500
data, labels = load_data_sign("../../resource/models/descriptors_pixel.pkl", "../../resource/models/labels.pkl")

model = get_pipe()
results = cross_val_score(model, data, labels.ravel(), cv=6)
print("Accuracy: " + str(mean(results)) + "(+/- " + str(std(results)))
