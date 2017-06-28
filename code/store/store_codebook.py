from sklearn.externals import joblib

from daq.dataset.fileaccess import get_paths_asl, read_image_asl
from preprocessing.bag_of_hogs import gen_codebook

paths_dict = get_paths_asl("../../resource/dataset/fingerspelling5/dataset5/", sets=["E"])
codebook_file = '../../resource/models/codebook.pkl'

images = []
for letter in paths_dict:
    for path in paths_dict[letter]:
        images.append(read_image_asl(path))
    break

codebook = gen_codebook(images, k_words=512)

joblib.dump(codebook, codebook_file)
