import cv2
import numpy as np

# from tracker.MeanShiftSegmentor import mean_shift_segmentation
from tracking.SkinSegmentor import filter_skin


def main():
    # Load an color image in colour
    dir_rsrc = '../../resource/dataset/'
    dir_a = 'fingerspelling5/dataset5/A/a/'
    test_a = 'color_0_0002.png'
    img = cv2.imread(dir_rsrc + dir_rsrc + dir_a + test_a, 1)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    skin = filter_skin(img)
    # show the skin in the image along with the mask
    cv2.imshow("detected skin", np.hstack([img, skin]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()