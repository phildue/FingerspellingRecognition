import cv2
import numpy as np
from sklearn.mixture import GMM, GaussianMixture

from daq.dataset.fileaccess import read_image, read_image_depth
from preprocessing.mrf import MarkovRandomField


def filter_skin(image):
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    # skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    # skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 2)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    return skin


example_image_file = "../../resource/dataset/fingerspelling5/dataset5/A/d/color_3_0007.png"
img = read_image(example_image_file)
img_depth = read_image_depth(example_image_file)
threshold = img_depth[int(img_depth.shape[0] / 2), int(img_depth.shape[1] / 2)]
cv2.imshow('depth', img_depth)
cv2.imshow('image', img)
label_grid = np.zeros(shape=(img.shape[0], img.shape[1], 2))

pixels = img[img_depth == threshold].reshape(-1, 3)

clusterer = GaussianMixture(n_components=2).fit(pixels)
label_fg = clusterer.predict_proba(pixels)

label_grid[img_depth == threshold] = np.uint8(label_fg * 255)

cv2.imshow('Soft labels', label_grid[:, :, 0])

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (11, 11), 5)
dx = cv2.Sobel(img_gray, -1, 0, 1, ksize=3)
dx = np.abs(dx)
dx[dx < 100] = 0
dy = cv2.Sobel(img_gray, -1, 1, 0, ksize=3)
dy = np.abs(dy)
dy[dy < 100] = 0

mrf = MarkovRandomField((img.shape[0], img.shape[1]), dx, dy, label_grid[:, :, 1], label_grid[:, :, 0])

segmented = mrf.maxflow()
cv2.imshow('Segmented', (segmented.reshape(img.shape[0], img.shape[1]) * 255).astype(np.uint8))

cv2.waitKey(0)

cv2.imshow('filtered', label_grid)
cv2.waitKey(0)
