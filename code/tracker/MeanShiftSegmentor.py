# Mean Shift Segmentation implementation
# Credits to http://www.chioka.in/meanshift-algorithm-for-the-rest-of-us-python/
import cv2
from numpy import array, zeros, linalg, exp, floor


class MeanShiftSegmentor:
    image = array
    kernel_size = int
    kernel = array

    def segment(self, image, kernel_size=3, bandwidth=16, max_iterations=100):
        self.image = image
        self.kernel_size = kernel_size
        self.kernel = self.__create_kernel(kernel_size, bandwidth)

        for i in range(0, max_iterations):
            x_max, y_max = self.image.shape
            image_prev = image
            for x in range(0, x_max):
                for y in range(y_max):
                    window = self.__neighbours(self.image[x, y])
                    self.image[x, y] = self.__weighted_sum(window)

            if linalg.norm(image_prev - image) < 0.0000001:
                break

        return image

    def __weighted_sum(self, window: array):
        weighted_sum = 0
        x_max, y_max = window.shape
        for x in range(0, x_max):
            for y in range(y_max):
                weighted_sum += self.kernel[x, y] * window[x, y]
        return weighted_sum

    def __neighbours(self, point):
        x = point(1)
        y = point(2)
        return self.image[x - 1:x + 1, y - 1:y + 1]

    def __create_kernel(self, kernel_size, bandwidth):
        mat = zeros(kernel_size)
        mean = zeros(2, 1)
        mean[:] = floor(kernel_size / 2)

        for x in range(0, kernel_size):
            for y in range(0, kernel_size):
                mat = self.__gauss([x, y], mean, bandwidth)

        return mat / linalg.norm(mat)

    def __gauss(self, point, mean, bandwidth) -> float:
        distance = mean - point
        return exp(-0.5 * (distance / bandwidth) ** 2)
