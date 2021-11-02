import cv2
from core.binarization import Binarization
import numpy as np

class SauvolaBinarization(Binarization):
    def __init__(self, window_size=31, R=128, k=0.2):
        self.window_size = window_size
        self.N = window_size * window_size
        self.half_window = window_size // 2
        self.R = R
        self.k = k

    def calc_integral_image(self, img):
        return np.cumsum(np.cumsum(img, axis=1), axis=0)

    def calc_rec(self, integral_img, x_1, y_1, x_2, y_2):

        d = integral_img[y_2, x_2]
        if x_1 != 0:
            c = integral_img[y_2, x_1 - 1]
        else:
            c = 0
        if y_1 != 0:
            b = integral_img[y_1 - 1, x_2]
        else:
            b = 0
        if x_1 != 0 and y_1 != 0:
            a = integral_img[y_1 - 1, x_1 - 1]
        else:
            a = 0
        return d - c - b + a

    def sliding_window(self, img):
        for x in range(img.shape[1] - self.half_window*2):
            for y in range(img.shape[0] - self.half_window*2):
                yield x, y, x + self.half_window*2, y + self.half_window*2

    def binarize(self):
        padded_img = np.pad(self.img, (self.half_window, self.half_window),mode='reflect')

        S_1 = self.calc_integral_image(padded_img)
        S_2 = self.calc_integral_image(np.power(padded_img, 2))

        windows = self.sliding_window(padded_img)

        for points in windows:

            rec_S_1 = self.calc_rec(S_1, *points)

            mean = rec_S_1/self.N

            std = np.sqrt(((self.calc_rec(S_2, *points) - np.power(rec_S_1,2)/self.N)/self.N))

            t = mean * (1 + self.k * (std/self.R - 1))

            img_point = (points[1], points[0])
            if self.img[img_point[0], img_point[1]] > t:
                self.img[img_point[0], img_point[1]] = 255
            else:
                self.img[img_point[0], img_point[1]] = 0






