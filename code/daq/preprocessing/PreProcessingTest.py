import cv2

# from tracker.MeanShiftSegmentor import mean_shift_segmentation
from daq.ImReader import read_im_file
from daq.preprocessing.PreProcessing import pre_processing


def main():
    # Load an color image in colour
    dir_rsrc = '../../../resource/dataset/'
    dir_a = 'fingerspelling5/dataset5/E/c/'
    sample_image = 'color_2_0192.png'
    img = read_im_file(dir_rsrc + dir_a + sample_image)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    img = pre_processing(img)
    cv2.imshow("after preprocessing", img)
    cv2.waitKey(10000)

    cv2.destroyAllWindows()


main()
