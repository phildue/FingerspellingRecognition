import random

import numpy as np

from classification.SignClassifier import SignClassifier
from daq.ImReader import read_image
from daq.preprocessing.PreProcessing import preprocess, extract_descriptor

# init
from representation.FeatureExtraction import extract_features

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]
example_image_file = "../../resource/dataset/tm/" + str(random.choice(letters)) + str(random.choice(range(1,40))) + ".tif"
dir_dataset = '../../resource/dataset/tm'
scaler_file = '../../resource/models/scaler.pkl'
extractor_file = '../../resource/models/extractor.pkl'
classif_file = '../../resource/models/classif.pkl'
cmd = 'y'
while cmd != 'n':
    # read image
    img = read_image(example_image_file)
    # crop hand
    img = preprocess(img)
    # extract descriptor
    descriptor = extract_descriptor(img)
    # apply preprocessing
    object_rep = extract_features(descriptor, scaler_file, extractor_file)
    # classify
    class_ = SignClassifier().classify(object_rep, classifier_file)
    # print output
    print("Letter " + str(letters[class_]))
    cmd = input("Continue? [y|n]: ")