import random

import cv2
from sklearn.externals import joblib

from datagen.FileProvider import FileProvider
from exceptions.exceptions import NoRoiFound
from preprocessing.PreProcessorTm import PreProcessorTm

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]

dir_dataset = '../../resource/dataset/tm'
model = joblib.load('../../resource/models/model_tm.pkl')
pp = PreProcessorTm(roi_size=(60, 60))
cmd = 'y'
while cmd != 'n':
    letter = str(random.choice(letters))
    example_image_file = "../../resource/dataset/tm/" + letter + str(
        random.choice(range(1, 40))) + ".tif"
    # read image
    img = FileProvider.read_img(example_image_file)
    #
    cv2.imshow("Read image", img)
    # crop hand
    try:
        img = pp.preprocess(img)
    except NoRoiFound:
        continue
    # extract descriptor
    cv2.imshow("Cropped hand", img)

    descriptor = pp.get_descr(img)

    # print(str(descriptor))
    # classify
    class_ = model.predict(descriptor)
    # print output
    print("Detected Letter " + letters[int(class_) - 1])
    print("Actual Letter: " + letter)
    cv2.waitKey(2000)

    # cmd = input("Continue? [y|n]: ")
