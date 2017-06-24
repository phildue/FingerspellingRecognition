import random

import cv2
from sklearn.externals import joblib

from daq.dataset.fileaccess import read_image, get_paths_tm
from daq.dataset.gendata import gendata_sign
from daq.preprocessing import preprocess, extract_descriptor
# init
from exceptions.exceptions import NoRoiFound

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]

dir_dataset = '../../resource/dataset/tm'
model = joblib.load('../../resource/models/model.pkl')
cmd = 'y'
while cmd != 'n':
    letter = str(random.choice(letters))
    example_image_file = "../../resource/dataset/tm/" + letter + str(
        random.choice(range(1, 40))) + ".tif"
    # read image
    img = read_image(example_image_file)
    #
    cv2.imshow("Read image", img)
    # crop hand
    try:
        img = preprocess(img)
    except NoRoiFound:
        continue
    # extract descriptor
    cv2.imshow("Cropped hand", img)

    descriptor = extract_descriptor(img)

    # print(str(descriptor))
    # classify
    class_ = model.predict(descriptor)
    # print output
    print("Detected Letter " + letters[int(class_) - 1])
    print("Actual Letter: " + letter)
    cv2.waitKey(2000)

    # cmd = input("Continue? [y|n]: ")
