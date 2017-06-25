import cv2
import maxflow as mf
import numpy as np


class MarkovRandomField:
    def __init__(self, img, smoothness, labels_soft):
        self.g = mf.Graph[int]()

        self.nodeids = self.g.add_grid_nodes(img.shape)

        self.g.add_grid_edges(self.nodeids, smoothness)
        self.g.add_grid_tedges(self.nodeids, labels_soft, 255 - labels_soft)

    def maxflow(self):
        self.g.maxflow()
        sgm = self.g.get_grid_segments(self.nodeids)
        labels_obtained = np.int_(np.logical_not(sgm))
        return labels_obtained * 255
