import numpy as np
import cv2 as cv
import os

# This and the other utils files are extremely dirty.
# They can be expected to break and/or cause irreparable data loss without warning.
# Use with caution or, better yet, not at all.

def main():
    folders = ['20200404/3', '20200405/1', '20200406/1', '20200407/1',
               '20200410/1', '20200410/DCA-ctrl', '20200410/DCA-0,15', '20200410/DCA-0,31', '20200410/DCA-0,62',
               '20200410/DCA-1,25', '20200410/DCA-2,50', '20200410/DCA-5,00',
               '20200408/1', '20200408/DCA-ctrl', '20200408/DCA-0,15', '20200408/DCA-0,31', '20200408/DCA-0,62', '20200408/DCA-1,25', '20200408/DCA-2,50', '20200408/DCA-5,00',
               '20200409/1', '20200409/DCA-ctrl', '20200409/DCA-0,15', '20200409/DCA-0,31', '20200409/DCA-0,62', '20200409/DCA-1,25', '20200409/DCA-2,50', '20200409/DCA-5,00',
               '20200411/1', '20200411/DCA-ctrl', '20200411/DCA-0,15', '20200411/DCA-0,31', '20200411/DCA-0,62', '20200411/DCA-1,25', '20200411/DCA-2,50', '20200411/DCA-5,00']

    circle_file = '/home/davidw/Desktop/circle_stats.csv'
    cf = open(circle_file, 'w')

    for ff in folders:
        print('Folder: {}'.format(ff))
        folder = os.path.join('/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/', ff, 'png_exports/full')
        files = sorted([file for file in os.listdir(folder) if os.path.splitext(file)[1] == '.png'])

        for f in files:
            c = find_circles(os.path.join(folder, f))
            if c is not None:
                cf.write('{}; {}; {}\n'.format(ff, f, [list(ci) for ci in c[0]]))

    cf.close()

def find_circles(im_file):
    img = cv.imread(im_file, cv.IMREAD_COLOR)
    img = cv.medianBlur(img, 7)
    g_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(g_img, cv.HOUGH_GRADIENT, 1, minDist=200,
                            param1=70, param2=50, minRadius=100,maxRadius=300)

    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
        # for i in circles[0,:]:
        #     # draw the outer circle
        #     cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #     # draw the center of the circle
        #     cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    # cv.imshow('detected circles', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return circles

main()