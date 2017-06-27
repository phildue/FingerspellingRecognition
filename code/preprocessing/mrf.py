import cv2
import maxflow as mf
import numpy as np


class MarkovRandomField:
    def __init__(self, shape, weights_x, weights_y, proba_foreground, proba_background):
        self.g = mf.Graph[int]()

        self.nodeids = self.g.add_grid_nodes(shape)

        self.g.add_grid_edges(self.nodeids, weights_y, structure=np.array([0, 0, 0,
                                                                           0, 0, 0,
                                                                           0, 1, 0]))

        self.g.add_grid_edges(self.nodeids, weights_x, structure=np.array([0, 0, 0,
                                                                           0, 0, 1,
                                                                           0, 0, 0]))

        self.g.add_grid_tedges(self.nodeids, proba_foreground, proba_background)

    def maxflow(self):
        self.g.maxflow()
        sgm = self.g.get_grid_segments(self.nodeids)
        labels_obtained = np.int_(np.logical_not(sgm))
        return labels_obtained
