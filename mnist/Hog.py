import numpy as np
import cv2
import config


def applymaskonimg(image, mask):
    return cv2.filter2D(image, cv2.CV_32F, mask)


def getimghog(image, maskx, masky, h, w, deg):
    if (config.height % h == 0) and (config.width % w == 0):
        gx = applymaskonimg(image, np.asmatrix(maskx).reshape((len(maskx), len(maskx))))
        gy = applymaskonimg(image, np.asmatrix(masky).reshape((len(masky), len(masky))))
        mag, ang = cv2.cartToPolar(gx, gy)

        # Creates array of degrees for Histogram in radians
        degrees = np.int32(deg * ang / (2 * np.pi))
        degrees = np.reshape(degrees, [config.height, config.width])
        mag = np.reshape(mag, [config.height, config.width])
        degree_cells = np.zeros([(config.height / h) * (config.width / w), h, w], 'int')
        mag_cells = np.zeros([(config.height / h) * (config.width / w), h, w], 'int')
        block = 0

        # Calculate cells
        for i in range(h, config.height + 1, h):
            for j in range(w, config.width + 1, w):
                starth = i - h
                startw = j - w

                degree_cells[block] = degrees[starth:i, startw:j]
                mag_cells[block] = mag[starth:i, startw:j]
                block += 1

        histogram = [np.bincount(b.ravel(), m.ravel(), minlength=deg) for b, m in zip(degree_cells, mag_cells)]
        return np.hstack(histogram).astype('float32')

    else:
        return None
