import random

from daq.fileaccess import get_paths_asl, read_image_asl
from preprocessing.preprocessing_asl import preprocess
from preprocessing.representation.BagOfHogs import BagOfHogs

codebook_file = '../../resource/models/codebook.pkl'

paths = get_paths_asl("../../resource/dataset/fingerspelling5/dataset5/")

paths = random.sample(paths[random.choice(list(paths.keys()))], k=100)

img = read_image_asl(random.choice(paths))
img = preprocess(img)

descriptor = BagOfHogs(codebook_file).get_descriptor(img)

print(str(descriptor))
