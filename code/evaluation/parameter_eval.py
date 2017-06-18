import numpy as np

from classification.SignClassifier import SignClassifier
from daq.DatasetGenerator import gendata_sign
from daq.ImReader import get_paths_tm
from evaluation.methods import crossval


def classifier_parameter_evaluation():
    results = np.zeros(20, 18)
    data, labels = gendata_sign(get_paths_tm())
    for n_sigma in range(-5, 15):
        for n_c in range(-15, 3):
            classifier = SignClassifier(gamma=pow(2, n_sigma), C=pow(2, n_c))
            results[n_sigma, n_c] = np.mean(crossval(classifier, data, labels))
            print("Results for sigma= " + str(pow(2, n_sigma)) + ", C= " + str(pow(2, n_c)) + ": " + str(
                results[n_sigma, n_c]))

    np.save("../../results/classifier_parameter_evaluation.npy", results)
