from sklearn.externals import joblib

from daq.fileaccess import get_paths_asl, read_image_asl
from preprocessing.preprocessing_asl import preprocesss
from preprocessing.representation.BagOfHogs import gen_codebook

paths_dict = get_paths_asl("../../resource/dataset/fingerspelling5/dataset5/")
codebook_file = '../../resource/models/codebook_total.pkl'

images = []
for letter in paths_dict:
    for path in paths_dict[letter]:
        images.append(read_image_asl(path))
    break

codebook = gen_codebook(preprocesss(images), k_words=512)

joblib.dump(codebook, codebook_file)
