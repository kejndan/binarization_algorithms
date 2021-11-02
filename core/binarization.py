import cv2
import matplotlib.pyplot as plt

class Binarization:
    def __init__(self):
        pass

    def read_img(self, path):
        self.img =  cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        return self.img

    def save_img(self, path):
        cv2.imwrite(path, self.img)

    def show_img(self):
        plt.imshow(self.img, cmap='gray')
        plt.axis(False)
        plt.show()


