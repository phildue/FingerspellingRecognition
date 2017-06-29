import random

from daq.fileaccess import get_paths_asl, read_image_asl
from preprocessing.bag_of_hogs import get_boh_descriptor
from preprocessing.preprocessing_asl import preprocess

codebook_file = '../../resource/models/codebook.pkl'

paths = get_paths_asl("../../resource/dataset/fingerspelling5/dataset5/")

paths = random.sample(paths[random.choice(list(paths.keys()))], k=100)

img = read_image_asl(random.choice(paths))
img = preprocess(img)

descriptor = get_boh_descriptor(img, codebook_file)

print(str(descriptor))
