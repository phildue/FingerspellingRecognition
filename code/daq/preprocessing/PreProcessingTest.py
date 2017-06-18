import cv2
import numpy as np

from daq.ImReader import get_paths_tm
from daq.preprocessing.PreProcessing import prefilter, extract_descriptor, get_longest_contours, get_centroid, \
    get_equally_distr_points, get_centroid_distances


def main():
    paths = get_paths_tm()
    img = cv2.imread(paths['b'][10])

    cv2.imshow('image', img)
    cv2.waitKey(5)
    img = prefilter(img)
    cv2.imshow("after prefiltering", img)
    cv2.waitKey(5)
    n_points = 1000
    img = prefilter(img, roi_size=(100, 100))
    contour = get_longest_contours(img)[0]
    centroid = get_centroid(contour)
    points = get_equally_distr_points(contour, n_points)
    point_img = np.zeros((100, 100, 3), np.uint8)
    for p in points:
        cv2.circle(point_img, (int(p[0]), int(p[1])), 1, (255, 255, 0))
    cv2.circle(point_img, (int(centroid[0]), int(centroid[1])), 1, (255, 0, 0))
    cv2.imshow("points along shape", point_img)
    cv2.waitKey(10000)

    descriptor = extract_descriptor(img, n_points)
    print("Descriptor: \n" + str(descriptor))
    print("dim: \n" + str(len(descriptor)))
    cv2.destroyAllWindows()
    exit(0)


main()
