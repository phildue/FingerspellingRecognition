import cv2

# from tracker.MeanShiftSegmentor import mean_shift_segmentation
from daq.datageneration.DatasetGeneratorSign import gendata_sign
from daq.preprocessing.PreProcessing import pre_processing


def main():

    img, _ = gendata_sign(dir_dataset='../../../resource/dataset/fingerspelling5/dataset5/',
                          sample_size=1,
                          sets=["E"])

    img.reshape(shape=(100, 120))

    cv2.imshow('image', img)
    cv2.waitKey(1)
    img = pre_processing(img)
    cv2.imshow("after preprocessing", img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    exit(0)


main()
