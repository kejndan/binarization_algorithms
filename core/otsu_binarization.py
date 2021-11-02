from core.binarization import Binarization
import numpy as np



class OtsuBinarization(Binarization):
    def __init__(self):
        super().__init__()

    def binarize(self):
        histogram, bins = np.histogram(self.img, 256, [0, 256])
        N = histogram.sum()
        thresholds = np.zeros(256)

        for k in range(256):
            iters_0 = np.arange(0, k)
            iters_1 = np.arange(k, 256)

            w_0 = histogram[:k].sum()/N
            w_1 = 1 - w_0

            m_0 = (histogram[:k]*iters_0).sum()/w_0 if w_0 != 0 else 0
            m_1 = (histogram[k:]*iters_1).sum()/w_1 if w_1 != 0 else 0

            t = w_0*w_1*(m_0 - m_1)**2
            thresholds[k] = t

        self.img = np.where(self.img > np.argmax(thresholds), 255, 0)
        print(np.argmax(thresholds))
        return self.img
