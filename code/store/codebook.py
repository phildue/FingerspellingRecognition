from sklearn.externals import joblib

from datagen.FileProviderAsl import FileProviderAsl
from preprocessing.PreProcessorAsl import PreProcessorAsl
from preprocessing.representation.BagOfHogs import BagOfHogs

paths_dict = FileProviderAsl("../../resource/dataset/fingerspelling5/dataset5/").img_file_paths
codebook_file = '../../resource/models/codebook_total.pkl'

images = []
for letter in paths_dict:
    for path in paths_dict[letter]:
        images.append(FileProviderAsl.read_img(path))
    break

codebook = BagOfHogs(n_bins=16, winsize=6).get_codebook(PreProcessorAsl(img_size=(60, 60)).preprocess_all(images),
                                                        k_words=512)

joblib.dump(codebook, codebook_file)
