import random

from daq.dataset.fileaccess import get_paths_asl, read_image_asl
from preprocessing.bag_of_hogs import gen_codebook, get_codebook_dist, get_boh_descriptor
from preprocessing.hog import get_hog
from preprocessing.preprocessing_asl import preprocess

codebook_file = '../../resource/models/codebook.pkl'

paths = get_paths_asl("../../resource/dataset/fingerspelling5/dataset5/")

paths = random.sample(paths[random.choice(list(paths.keys()))], k=100)

img = read_image_asl(random.choice(paths))
img = preprocess(img)

descriptor = get_boh_descriptor(img, codebook_file)

print(str(descriptor))
