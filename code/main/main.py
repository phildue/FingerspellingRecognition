import random

import numpy as np

from classification.SignClassifier import SignClassifier
from daq.ImReader import read_image
from daq.preprocessing.PreProcessing import prefilter, extract_descriptor

# init
from representation.FeatureExtraction import extract_features

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]
example_image_file = "../resource/dataset/tm/" + random.choice(letters) + random.choice(range(1,40)) + ".gif"
classifier_file = "../resource/models/classif.mdl"
scaler_file = "../resource/models/scaler.mdl"
extractor_file = "../resource/models/extractor.mdl"
cmd = 'y'
while cmd != 'n':
    # read image
    img = read_image(example_image_file)
    # crop hand
    img = prefilter(img)
    # extract descriptor
    descriptor = extract_descriptor(img)
    # apply preprocessing
    object_rep = extract_features(descriptor, scaler_file, extractor_file)
    # classify
    class_ = SignClassifier().classify(object_rep, classifier_file)
    # print output
    print("Letter " + str(letters[class_]))
    cmd = input("Continue? [y|n]: ")