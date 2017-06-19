import pickle

from classification.SignClassifier import SignClassifier
from daq.DatasetGenerator import gendata_sign
from daq.ImReader import get_paths_tm
from representation.FeatureExtraction import get_scaler, get_extractor


def train_models(dir_dataset='../../resource/dataset/tm',
                 scaler_file='../../resource/models/scaler.mdl',
                 extractor_file='../../resource/models/extractor.mdl',
                 classif_file='../../resource/models/classif.mdl'):
    # Reads dataset, trains models and stores them on file system
    data, labels = gendata_sign(get_paths_tm(dir_dataset=dir_dataset), 2000)
    scaler = get_scaler(data)
    extractor = get_extractor(data)
    data = scaler.transform(data)
    data = extractor.transform(data)
    classif = SignClassifier()
    classif.train(data, labels)
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    with open(extractor_file, 'wb') as f:
        pickle.dump(extractor, f)
    with open(classif_file, 'wb') as f:
        pickle.dump(classif, f)

train_models()