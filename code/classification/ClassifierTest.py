from classification.ClassifierClass import Classifier
from evaluation.hold_out import hold_out_eval
from feature_generation.DataGenerator import gendata
from feature_generation.FeatureExtraction import pca_transform


def main():
    dir_dataset = '../../resource/dataset/fingerspelling5/dataset5/A/'
    n_data = 100
    data, labels = gendata(dir_dataset, n_data)
    data = pca_transform(data)
    error = hold_out_eval(Classifier(), data, labels)

    print("Error: " + str(error))


main()
