import random

from datagen.FileProviderAsl import FileProviderAsl
from preprocessing.PreProcessorAsl import PreProcessorAsl
from preprocessing.representation.BagOfHogs import BagOfHogs
from preprocessing.segmentation.MRFAsl import MRFAsl

codebook_file = '../../resource/models/codebook.pkl'

paths = FileProviderAsl("../../resource/dataset/fingerspelling5/dataset5/").img_file_paths

paths = random.sample(paths[random.choice(list(paths.keys()))], k=100)

img = FileProviderAsl.read_img(random.choice(paths))
pp = PreProcessorAsl(segmenter=MRFAsl(), descriptor=BagOfHogs(codebook_file))
img = pp.preprocess(img)

descriptor = pp.get_descr(img)

print(str(descriptor))
