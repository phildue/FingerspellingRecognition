import random

from daq.dataset.fileaccess import get_paths_asl, read_image_asl
from preprocessing.bag_of_hogs import gen_codebook, get_codebook_dist
from preprocessing.hog import get_hog
from preprocessing.preprocessing_asl import preprocess

paths = get_paths_asl("../../resource/dataset/fingerspelling5/dataset5/")

paths = random.sample(paths[random.choice(list(paths.keys()))], k=100)

img = read_image_asl(random.choice(paths))
img = preprocess(img)

codebook = gen_codebook(paths, k_words=10)

print(str(codebook))

print(str(get_codebook_dist(get_hog(img), codebook)))
