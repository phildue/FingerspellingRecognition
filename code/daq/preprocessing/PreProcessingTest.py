import cv2

# from tracker.MeanShiftSegmentor import mean_shift_segmentation
from daq.ImReader import read_im_file
from daq.preprocessing.SkinSegmentor import filter_skin


def main():
    # Load an color image in colour
    dir_rsrc = '../../../resource/dataset/'
    dir_a = 'fingerspelling5/dataset5/A/c/'
    sample_image = 'color_2_0002.png'
    img = read_im_file(dir_rsrc + dir_a + sample_image)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    skin = filter_skin(img)
    cv2.imshow("skin", skin)
    cv2.waitKey(10000)
    skin = cv2.cvtColor(skin, cv2.COLOR_RGB2GRAY)
    ## Reduce
    skin = cv2.blur(skin, (7, 7))
    ## Canny
    skin = cv2.Canny(skin, threshold1=50, threshold2=150)
    _, skin = cv2.threshold(skin, 50, 255, cv2.THRESH_BINARY)
    # show the skin in the image along with the mask
    # cv2.imshow("detected skin", np.hstack([img, skin]))
    cv2.imshow("skin binary ", skin)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


main()
