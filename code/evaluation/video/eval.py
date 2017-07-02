import numpy as np

from classification.EstimatorVideo import EstimatorVideo
from datagen.FileProvider import FileProvider
from evaluation.video.BackgroundSubtractorDummy import BackgroundSubtractorDummy
from preprocessing.PreProcessorVideo import PreProcessorVideo
from preprocessing.representation.HistogramOfGradients import HistogramOfGradients

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]

test_image_path = "../../../resource/dataset/camtest/"
estimator = EstimatorVideo(model_path='../../../resource/models/model_hog_asl.pkl')
predictions = []
score = 0
for letter_i, letter in enumerate(letters):
    for i in range(1, 5):
        diff_image = FileProvider.read_img(test_image_path + letter + '/Difference' + str(i) + ".jpg")
        test_image = FileProvider.read_img(test_image_path + letter + '/Original' + str(i) + ".jpg")

        if i == 1:
            preprocessor = PreProcessorVideo(background_subtractor=BackgroundSubtractorDummy(diff_image, test_image),descriptor=HistogramOfGradients(n_bins=16))
            preprocessor.calibrate_object(test_image)

        test_image = preprocessor.preprocess(test_image)
        descr = preprocessor.get_descr(test_image)
        estimator.stack_descr(descr)

    class_ = estimator.predict()
    predictions.append(class_-1)

    if letter is letters[class_-1]:
       score+=1

accuracy = score/24

print("Accuracy: " + str(accuracy))
print(str(predictions))