from numpy import mean, std
from sklearn.model_selection import cross_val_score

from classification.pipe import get_pipe
from datagen.DatasetGenerator import DatasetGenerator
from datagen.FileProviderAsl import FileProviderAsl
from preprocessing.PreProcessorAsl import PreProcessorAsl
from preprocessing.representation.BagOfHogs import BagOfHogs

g = DatasetGenerator(FileProviderAsl("../../../resource/dataset/fingerspelling5/dataset5/"),
                     preprocessor=PreProcessorAsl(
                         descriptor=BagOfHogs("../../../resource/models/codebook_total_new.pkl")))
data, labels = g.generate()

model = get_pipe()
results = cross_val_score(model, data, labels.ravel(), cv=6)
print("Accuracy: " + str(mean(results)) + "(+/- " + str(std(results)))
