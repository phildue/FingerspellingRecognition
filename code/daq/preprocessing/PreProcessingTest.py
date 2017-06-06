import cv2

# from tracker.MeanShiftSegmentor import mean_shift_segmentation
from daq.ImReader import read_im_file
from daq.preprocessing.PreProcessing import pre_processing
from daq.preprocessing.SkinSegmentor import filter_skin


def main():
    # Load an color image in colour
    dir_rsrc = '../../../resource/dataset/'
    dir_a = 'fingerspelling5/dataset5/E/c/'
    sample_image = 'color_2_0192.png'
    img = read_im_file(dir_rsrc + dir_a + sample_image)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    img = pre_processing(img)
    cv2.imshow("skin", img)
    cv2.waitKey(10000)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # ## Reduce
    img = cv2.blur(img, (7, 7))
    # ## Canny
    img = cv2.Canny(img, threshold1=20, threshold2=100)
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    # show the skin in the image along with the mask
    # cv2.imshow("detected skin", np.hstack([img, skin]))
    cv2.imshow("skin binary ", img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


main()
