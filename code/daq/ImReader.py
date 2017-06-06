import cv2


def read_im_file(path):
    img = cv2.imread(path, 1)
    if img is None:
        raise FileNotFoundError("Couldn't find: " + path)
    return img
