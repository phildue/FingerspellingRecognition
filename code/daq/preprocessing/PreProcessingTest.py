import cv2

from daq.ImReader import get_paths_tm
from daq.preprocessing.PreProcessing import prefilter, extract_descriptor


def main():
    paths = get_paths_tm()
    img = cv2.imread(paths['b'][10])

    cv2.imshow('image', img)
    cv2.waitKey(5)
    img = prefilter(img)
    cv2.imshow("after prefiltering", img)
    cv2.waitKey(10000)

    descriptor = extract_descriptor(img)
    print("Descriptor: \n" + str(descriptor))
    print("dim: \n" + str(len(descriptor)))
    cv2.destroyAllWindows()
    exit(0)


main()
