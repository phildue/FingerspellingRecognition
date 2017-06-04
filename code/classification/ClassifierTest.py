from classification.ClassifierClass import Classifier
from evaluation.HoldOut import hold_out_eval
from featuregeneration.DataGenerator import gendata
from featuregeneration.DissimilarityRep import get_dissim_rep
from featuregeneration.FeatureExtraction import pca_transform


def main():
    dir_dataset = '../../resource/dataset/fingerspelling5/dataset5/'
    n_data = 10000
#    data, labels = gendata(dir_dataset, n_data, alphabet=["a", "b"])
    data, labels = gendata(dir_dataset, n_data)

    data = pca_transform(data)
    # data = get_dissim_rep(data)

    error = hold_out_eval(Classifier(), data, labels)

    print("Error: " + str(error))


main()
