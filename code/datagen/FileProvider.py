import random
from abc import abstractmethod

import cv2


class FileProvider:
    def __init__(self, paths=None):
        self.img_file_paths = paths

    def read_letters(self, sample_size, letters):
        letters_random_order = letters.copy()
        random.shuffle(letters_random_order)
        letter_imgs = {}
        for letter in letters_random_order:
            letter_imgs[letter] = []
            try:
                path_sel = random.sample(self.img_file_paths[letter], sample_size)
            except ValueError:
                path_sel = self.img_file_paths[letter]
                print("Too many samples requested for [" + letter + "], taking all: " + str(len(path_sel)))

            for sel_samples, path in enumerate(path_sel):
                img = self.read_img(path)
                letter_imgs[letter].append(img)

        return letter_imgs

    @staticmethod
    def read_img(path):
        img = None
        try:
            img = cv2.imread(path, 1)
            if img is None:
                raise FileNotFoundError("Couldn't find: " + path)
        except Exception:
            print("Exception on reading file :" + str(path))

        return img
