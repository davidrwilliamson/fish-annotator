import cv2 as cv
import numpy as np


def find_circles(im_file):
    im = np.load(im_file).astype('uint8').squeeze()
    im = cv.cvtColor(im, cv.COLOR_BAYER_BG2RGB)
    im = cv.medianBlur(im, 7)
    g_im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(g_im, cv.HOUGH_GRADIENT, 1, minDist=200,
                              param1=70, param2=50, minRadius=100, maxRadius=300)

    return circles
