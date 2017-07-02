import os

from daq.FileProvider import FileProvider


class FileProviderAsl(FileProvider):
    def __init__(self, dir_dataset='../../../resource/dataset/fingerspelling5/dataset5/', sets=None, alphabet=None):
        FileProvider.__init__(self)
        if alphabet is None:
            alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                        "u",
                        "v", "w", "x", "y"]
        if sets is None:
            sets = ["A", "B", "C", "D", "E"]

        paths = {}

        for dir_letter in alphabet:
            paths[dir_letter] = []
            for dir_set in sets:
                fnames = [f for f in os.listdir(dir_dataset + dir_set + "/" + dir_letter) if 'color' in f]
                for fname in fnames:
                    paths[dir_letter].append(dir_dataset + dir_set + "/" + dir_letter + "/" + fname)

        self.img_file_paths = paths

    @staticmethod
    def read_img(path):
        return FileProviderAsl.read_img_colour(path), FileProviderAsl.read_image_depth(path)

    @staticmethod
    def read_img_colour(path):
        img = None
        try:
            img = cv2.imread(path, 1)
            if img is None:
                raise FileNotFoundError("Couldn't find: " + path)
        except Exception:
            print("Exception on reading file :" + str(path))

        return img

    @staticmethod
    def read_image_depth(path):
        path_depth = path.replace("color", "depth")
        try:
            img = cv2.imread(path_depth, 0)
            if img is None:
                raise FileNotFoundError("Couldn't find: " + path_depth)
        except Exception:
            print("Exception on reading file :" + str(path_depth))

        return img
