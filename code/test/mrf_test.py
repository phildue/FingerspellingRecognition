import cv2
import maxflow as mf
import numpy as np

from app.models.HistSegmenter import HistSegmenter
from preprocessing.mrf import MarkovRandomField


def example():
    g = mf.Graph[int]()
    img = cv2.imread("../../resource/a2.png")

    cv2.imshow("Original", img)
    cv2.waitKey(0)

    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    mrf = MarkovRandomField(img, 180, img)

    img2 = mrf.maxflow()

    _, img2 = cv2.threshold(img2.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)

    cv2.imshow("Denoised", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("../../resource/dataset/fingerspelling5/dataset5/A/a/color_0_0004.png")

cv2.imshow("Picture", img)
likelihood = img.copy()
histmodel = HistSegmenter("../../resource/models/skinhist_asl.npy")
winsize = 7
step = 3
for y in range(0, img.shape[0] - winsize + 1, step):
    for x in range(0, img.shape[1] - winsize + 1, step):
        likelihood[y:y + winsize, x: x + winsize] = histmodel.get_likelihood(
            img[y:y + winsize, x:x + winsize].reshape(-1, 3), sigma=550) * 255
cv2.normalize(likelihood, likelihood, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("Likelihood", likelihood)
cv2.waitKey(0)
# TODO try smoothening with derivatives, add weights towards center of the image
mrf = MarkovRandomField(img, 180, likelihood)
img2 = mrf.maxflow()

_, img2 = cv2.threshold(img2.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)

cv2.imshow("Denoised", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
