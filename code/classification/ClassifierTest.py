from classification.SignClassifier import SignClassifier
from classification.SkinClassifier import SkinClassifier
from daq.DatasetGenerator import gendata_sign, gendata_skin
from daq.ImReader import get_paths_tm
from evaluation.HoldOut import hold_out_eval
from representation.DissimilarityRep import get_dissim_rep
from representation.FeatureExtraction import pca_transform


def SignClassifierTest():
    dir_dataset = '../../resource/dataset/fingerspelling5/dataset5/'
    n_data = 60
    #    data, labels = gendata(dir_dataset, n_data, alphabet=["a", "b"])
    data, labels = gendata_sign(get_paths_tm(dir_dataset='../../resource/dataset/tm'
                                             ), n_data)

    data = pca_transform(data)
    # data = get_dissim_rep(data)

    error = hold_out_eval(SignClassifier(), data, labels)

    print("Error: " + str(error))


def SkinClassifierTest():
    data, labels = gendata_skin()

    error = hold_out_eval(SkinClassifier(), data, labels)

    print("Error: " + str(error))


SignClassifierTest()
# SkinClassifierTest()
