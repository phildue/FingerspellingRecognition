import random

import cv2
from sklearn.externals import joblib

from daq.ImReader import read_image
from daq.preprocessing.PreProcessing import preprocess, extract_descriptor

# init

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]

dir_dataset = '../../resource/dataset/tm'
model = joblib.load('../../resource/models/model.pkl')

cmd = 'y'
while cmd != 'n':
    example_image_file = "../../resource/dataset/tm/" + str(random.choice(letters)) + str(
        random.choice(range(1, 40))) + ".tif"
    # read image
    img = read_image(example_image_file)
    #
    cv2.imshow("Read image",img)
    # crop hand
    img = preprocess(img)
    # extract descriptor
    descriptor = extract_descriptor(img)
    # classify
    class_ = model.predict(descriptor)
    # print output
    print("Letter " + str(letters[int(class_)]))
    cv2.waitKey(0)

    # cmd = input("Continue? [y|n]: ")
