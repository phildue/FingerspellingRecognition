import numpy as np
from daq.datageneration.ImReader import get_paths_tm
from representation.FeatureExtraction import pca_fit_transform

from classification.pipe import SignClassifier
from daq.gendata import gendata_sign
from evaluation.methods import crossval


def classifier_parameter_evaluation():
    results = np.zeros(shape=(20, 18))
    data, labels = gendata_sign(get_paths_tm("../../resource/dataset/tm"))
    for n_sigma in range(-5, 15):
        for n_c in range(-15, 3):
            classifier = SignClassifier(gamma=pow(2, n_sigma), C=pow(2, n_c))
            results[n_sigma, n_c] = np.mean(crossval(classifier, data, labels))
            print("Results for sigma= " + str(pow(2, n_sigma)) + ", C= " + str(pow(2, n_c)) + ": " + str(
                results[n_sigma, n_c]))

    np.save("../../results/classifier_parameter_evaluation.npy", results)


def classifier_C_evaluation():
    results = np.zeros(shape=(13, 2))
    data, labels = gendata_sign(get_paths_tm("../../resource/dataset/tm"))
    data = pca_fit_transform(data)
    for i, n_c in enumerate(range(-3, 10)):
        classifier = SignClassifier(C=pow(2, n_c))
        result = crossval(classifier, data, labels, 3)
        results[i, 0] = np.mean(result)
        results[i, 1] = np.std(result)
        print("Accuracy for sigma= " + "auto" + ", C= " + str(pow(2, n_c)) + ": " + str(results[i, 0]) + "(+/- "+str(results[i,1])+ ")")

    np.save("../../results/classifier_C_evaluation.npy", results)


classifier_C_evaluation()
