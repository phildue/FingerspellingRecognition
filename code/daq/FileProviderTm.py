import os

from daq.FileProvider import FileProvider


class FileProviderTm(FileProvider):
    def __init__(self, dir_dataset='../../../resource/dataset/tm'):
        FileProvider.__init__(self)
        alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                    "v", "w", "x", "y"]
        paths = {}

        for dir_letter in alphabet:
            paths[dir_letter] = []
            i = 1
            while os.path.isfile(dir_dataset + '/' + dir_letter + str(i) + '.tif'):
                paths[dir_letter].append(dir_dataset + '/' + dir_letter + str(i) + '.tif')
                i += 1

        self.img_file_paths = paths
