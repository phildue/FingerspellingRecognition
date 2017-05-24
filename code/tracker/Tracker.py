import numpy as np
import cv2


def mean_shift_segmentation(img):
    return cv2.pyrMeanShiftFiltering(img, 32, 32)

def main():
    # Load an color image in grayscale
    dir_rsrc = '../../resource/dataset/'
    dir_a = 'fingerspelling5/dataset5/A/a/'
    test_a = 'color_0_0002.png'
    img = cv2.imread(dir_rsrc + dir_rsrc + dir_a + test_a, 1)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    img = mean_shift_segmentation(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()