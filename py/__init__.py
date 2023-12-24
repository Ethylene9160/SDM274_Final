import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# import cppyy
# import ctypes

class Hopfield:
    def __init__(self, size, iter=10):
        self.iter = iter
        self.size = size
        self.W = np.zeros((size ** 2, size ** 2))


    def train(self, X):
        n = self.size ** 2
        for x in X:  # (-1,64*64)
            x = np.reshape(x, (n, 1))
            xT = np.reshape(x, (1, n))
            self.W += x * xT / n
        self.W[np.diag_indices_from(self.W)] = 0

    def test_one_frame(self, x):
        n = self.size ** 2
        x = np.reshape(x, (n,))
        energy = []
        for iter in range(self.iter):
            h = np.zeros((n,))
            for i in range(n):
                i = np.random.randint(n)
                h[i] = self.W[i, :].dot(x)
            x[h > 0] = 1
            x[h < 0] = -1
        energy.append(self.cal_energy(x))

        return np.resize(x, (self.size, self.size)), energy

    def cal_energy(self, x):
        n = self.size ** 2
        energy = np.sum(self.W.dot(x) * x)

        return -0.5 * energy

    def nextX(self):
        pass


def show(x):
    img = np.where(x > 0, 255, 0).astype(np.uint8)
    cv.imshow("img", img)
    cv.waitKey(0)


if __name__ == "__main__":

    img = cv.imread(r"D:\resource\mizh\kazuma.jpg", 0)
    size = 128
    img = cv.resize(img, (size, size))
    x = np.where(img > 255 / 2.5, 1, -1)
    x_masked = x.copy()
    x_masked[size // 2:, :] = -1
    # show(x_masked)

    plt.figure(figsize=(9, 9))
    plt.subplot(2,2,1)
    plt.imshow(x)
    plt.title('Kazuha')

    plt.subplot(2,2,2)
    plt.imshow(x_masked)
    plt.title('Masked Kazuha')

    model = Hopfield(size, iter=3)
    # x1 = [x]
    # x1 = x1.copy
    model.train([x])
    y, energy = model.test_one_frame(x_masked)
    # show(y)
    plt.subplot(2,2,3)
    plt.imshow(y)
    plt.title('Recovered Kazuha (epochs = 3)')

    model2 = Hopfield(size, iter = 10)
    plt.subplot(2,2,4)
    model2.train([x])
    y2, energy2 = model2.test_one_frame(x_masked)
    plt.imshow(y2)
    plt.title('Recovered Kazuha (epochs = 10)')
    plt.show()

    # plt.plot(energy, label='energy')
    # plt.show()
