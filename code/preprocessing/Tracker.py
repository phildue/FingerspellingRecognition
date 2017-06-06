import cv2
import numpy as np

# from tracker.MeanShiftSegmentor import mean_shift_segmentation
from featuregeneration.DataGenerator import gendata
from preprocessing.SkinSegmentor import filter_skin


def main():
    # Load an color image in colour
    dir_rsrc = '../../resource/dataset/'
    dir_a = 'fingerspelling5/dataset5/A/c/'
    test_a = 'color_2_0135.png'
    img = cv2.imread(dir_rsrc + dir_rsrc + dir_a + test_a, 1)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    skin = filter_skin(img)
    skin = cv2.cvtColor(skin, cv2.COLOR_RGB2GRAY)
    # show the skin in the image along with the mask
    # cv2.imshow("detected skin", np.hstack([img, skin]))
    cv2.imshow("skin binary ", skin)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


main()