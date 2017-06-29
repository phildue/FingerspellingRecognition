import cv2

from app.models.HistSegmenter import HistSegmenter

img = cv2.imread("../../resource/dataset/fingerspelling5/dataset5/A/a/color_0_0004.png")

cv2.imshow("Picture", img)
likelihood = img.copy()
histmodel = HistSegmenter("../../resource/models/skinhist_asl.npy")
winsize = 7

for y in range(0, img.shape[0] - winsize + 1, 3):
    for x in range(0, img.shape[1] - winsize + 1, 3):
        likelihood[y:y + winsize, x: x + winsize] = histmodel.get_likelihood(
            img[y:y + winsize, x:x + winsize].reshape(-1, 3), sigma=500) * 255
cv2.normalize(likelihood, likelihood, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("Likelihood", likelihood)
cv2.waitKey(0)
