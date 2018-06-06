import cv2 as cv
import numpy as np
import time
from skimage import color
from console_progressbar import ProgressBar

if __name__ == '__main__':
    print('SkImage:')
    start = time.time()
    L = [0] * 256 ** 3
    a = [0] * 256 ** 3
    b = [0] * 256 ** 3
    i = 0
    pb = ProgressBar(total=256, prefix='SkImage converting images', suffix='', decimals=3, length=50, fill='=')
    for r in range(256):
        for g in range(256):
            for bb in range(256):
                im = np.array((bb, g, r), np.uint8).reshape(1, 1, 3)
                color.rgb2lab(im)  # transform it to LAB
                L[i] = im[0, 0, 0]
                a[i] = im[0, 0, 1]
                b[i] = im[0, 0, 2]
                i += 1
        pb.print_progress_bar(r)

    print(min(L), '<=L<=', max(L))
    print(min(a), '<=a<=', max(a))
    print(min(b), '<=b<=', max(b))
    end = time.time()
    elapsed = end - start
    print('elapsed: {} seconds.'.format(elapsed))

    print('OpenCV:')
    start = time.time()
    L = [0] * 256 ** 3
    a = [0] * 256 ** 3
    b = [0] * 256 ** 3
    i = 0
    pb = ProgressBar(total=256, prefix='OpenCV converting images', suffix='', decimals=3, length=50, fill='=')
    for r in range(256):
        for g in range(256):
            for bb in range(256):
                im = np.array((bb, g, r), np.uint8).reshape(1, 1, 3)
                cv.cvtColor(im, cv.COLOR_BGR2LAB, im)  # transform it to LAB
                L[i] = im[0, 0, 0]
                a[i] = im[0, 0, 1]
                b[i] = im[0, 0, 2]
                i += 1
        pb.print_progress_bar(r)

    print(min(L), '<=L<=', max(L))
    print(min(a), '<=a<=', max(a))
    print(min(b), '<=b<=', max(b))
    end = time.time()
    elapsed = end - start
    print('elapsed: {} seconds.'.format(elapsed))
