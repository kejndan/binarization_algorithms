import cv2
import matplotlib.pyplot as plt
import numpy as np


class Binarization:
    def __init__(self):
        pass

    def read_img(self, path):
        self.img =  cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY).astype(np.int64)

        return self.img

    def save_img(self, path):
        cv2.imwrite(path, self.img)

    def show_img(self, title=None):
        plt.imshow(self.img, cmap='gray')
        plt.axis(False)
        if title is not None:
            plt.title(title)
        plt.show()


