import cv2
import maxflow as mf
import numpy as np


def example():
    g = mf.Graph[int]()
    img = cv2.imread("../../resource/a2.png")
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    nodeids = g.add_grid_nodes(img.shape)
    weight = 180
    cv2.imshow("Original", img)
    cv2.waitKey(0)

    g.add_grid_edges(nodeids, weight)
    g.add_grid_tedges(nodeids, img, 255 - img)

    g.maxflow()
    sgm = g.get_grid_segments(nodeids)
    img2 = np.int_(np.logical_not(sgm))
    _, img2 = cv2.threshold(img2.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)

    cv2.imshow("Denoised", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("../../resource/dataset/fingerspelling5/dataset5/A/a/color_0_0002.png")
