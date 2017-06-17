import random

import cv2

# from tracker.MeanShiftSegmentor import mean_shift_segmentation
from daq.DatasetGenerator import gendata_sign
from daq.ImReader import get_paths_tm
from daq.preprocessing.PreProcessing import prefilter


def main():

    paths = get_paths_tm()
    img = cv2.imread(paths['b'][2])

    cv2.imshow('image', img)
    cv2.waitKey(1)
    img = prefilter(img)
    cv2.imshow("after preprocessing", img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    exit(0)


main()
