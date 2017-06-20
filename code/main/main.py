import random

import cv2
from sklearn.externals import joblib

from daq.dataset.fileaccess import read_image
from daq.dataset.preprocessing import preprocess, extract_descriptor
# init
from exceptions.exceptions import NoRoiFound

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]

dir_dataset = '../../resource/dataset/tm'
model = joblib.load('../../resource/models/model.pkl')
cap = cv2.VideoCapture()
if not cap.isOpened:
    print("Could not read webcam")

while cap.isOpened and cv2.waitKey(10) != 27:
    # letter = str(random.choice(letters))
    # example_image_file = "../../resource/dataset/tm/" + letter + str(
    #    random.choice(range(1, 40))) + ".tif"
    # read image
    # img = read_image(example_image_file)
    success, img = cap.read()
    if not success:
        print("Error on reading webcam")
        break

    cv2.imshow("Read image", img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #
    # crop hand
    try:
        img = preprocess(img)
    except NoRoiFound:
        continue
    # extract descriptor
    descriptor = extract_descriptor(img)
    # classify
    class_ = model.predict(descriptor)
    # print output
    print("Detected Letter " + str(letters[int(class_) - 1]))
    print("Actual Letter: " + letter)

cv2.destroyAllWindows()
