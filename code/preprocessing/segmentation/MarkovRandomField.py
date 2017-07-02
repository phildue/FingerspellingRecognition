from abc import abstractmethod

import cv2
import maxflow as mf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from preprocessing.segmentation.Segmenter import Segmenter


class MarkovRandomField(Segmenter):
    @abstractmethod
    def get_label(self, img):
        pass

    def get_label_soft(self, img):
        print("Warning! MarkovRandomField doesnt provides soft labels! ")
        return self.get_label(img)

    @staticmethod
    def maxflow(graph, nodeids):
        graph.maxflow()

        sgm = graph.get_grid_segments(nodeids)
        return np.int_(np.logical_not(sgm))

    @staticmethod
    def create_graph(shape, weight_x, weight_y, likelihood_object, likelihood_backgr):
        g = mf.Graph[int]()

        nodeids = g.add_grid_nodes(shape)

        g.add_grid_edges(nodeids, weight_y, structure=np.array([0, 1, 0,
                                                                0, 0, 0,
                                                                0, 1, 0]))

        g.add_grid_edges(nodeids, weight_x, structure=np.array([0, 0, 0,
                                                                1, 0, 1,
                                                                0, 0, 0]))

        g.add_grid_tedges(nodeids, likelihood_object, likelihood_backgr)

        return g, nodeids

    def get_smooth_grid(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.normalize(img_gray.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX)
        img_gray = cv2.GaussianBlur(img_gray, (11, 11), 5)
        dx = cv2.Sobel(img_gray, -1, 0, 1, ksize=3)
        dx = np.abs(dx)
        dx[dx < 20] = 0
        dy = cv2.Sobel(img_gray, -1, 1, 0, ksize=3)
        dy = np.abs(dy)
        dy[dy < 20] = 0
        wx = -cv2.normalize(dx.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX)
        wy = -cv2.normalize(dy.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX)
        return wx, wy

    def get_background_score(self, img, background, bandwidth=10):
        background_score_grid = np.zeros(shape=(background.shape[0], background.shape[1], 2))
        background_score_grid[:, :, 1] = np.exp(-np.abs(img.astype('float') - background.astype('float')) / bandwidth)
        background_score_grid[:, :, 0] = 1 - background_score_grid[:, :, 0]

        return background_score_grid

    def get_classifier_score(self, img, pixels_fg, pixels_bg):
        pixels_fg = pixels_fg[np.random.randint(pixels_fg.shape[0], size=200), :]
        pixels_bg = pixels_bg[np.random.randint(pixels_bg.shape[0], size=1000), :]

        data = np.vstack([pixels_bg, pixels_fg])
        labels = np.vstack([np.zeros(shape=(pixels_bg.shape[0], 1)), np.ones(shape=(pixels_fg.shape[0], 1))])

        knn = KNeighborsClassifier(3).fit(data, labels.ravel())

        return knn.predict_proba(img.reshape(-1, 3)).reshape(img.shape[0], img.shape[1], 2)

    def get_weighted_sum(self, classifier_score_grid, background_score_grid, w_classifier=0.7):
        w_background = 1 - w_classifier
        score_grid = w_classifier * classifier_score_grid + w_background * background_score_grid

        return cv2.normalize(score_grid.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX)
