import numpy as np

from classification.EstimatorVideo import EstimatorVideo
from datagen.FileProvider import FileProvider
from evaluation.video.BackgroundSubtractorDummy import BackgroundSubtractorDummy
from preprocessing.PreProcessorVideo import PreProcessorVideo

letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
           "u",
           "v", "w", "x", "y"]

test_image_path = "../../"
estimator = EstimatorVideo(model_path='../../resource/models/model_hog_asl.pkl')
confusion_matrix = np.zeros(shape=(24, 24))

for letter_i, letter in enumerate(letters):
    frames = []
    for frame in frames:
        diff_image = FileProvider.read_img(p)
        test_image = FileProvider.read_img(p)

        preprocessor = PreProcessorVideo(background_subtractor=BackgroundSubtractorDummy(diff_image, test_image))
        test_image = preprocessor.preprocess(test_image)
        descr = preprocessor.get_descr(test_image)
        estimator.stack_descr(descr)

    confusion_matrix[letter_i, estimator.predict()] += 1

print(str(confusion_matrix))
