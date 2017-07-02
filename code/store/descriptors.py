from datagen.DatasetGenerator import DatasetGenerator
from datagen.FileProviderAsl import FileProviderAsl
from preprocessing.PreProcessorAsl import PreProcessorAsl
from preprocessing.representation.HistogramOfGradients import HistogramOfGradients
from preprocessing.segmentation.MRFAsl import MRFAsl

g = DatasetGenerator(file_provider=FileProviderAsl("../../resource/dataset/fingerspelling5/dataset5/"),
                     preprocessor=PreProcessorAsl(segmenter=MRFAsl(),
                                                  descriptor=HistogramOfGradients(window_size=6, n_bins=8)))

data, labels = g.generate(sample_size=2500)

g.save(data, labels, '../../resource/models/', 'hog')
