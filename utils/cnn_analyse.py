import cv2 as cv
import numpy as np
import os

root_folder = '/media/dave/My Passport/Easter/Bernard/'
date_folder = '20200415'
treatment_folders = ['DCA-0,15', 'DCA-0,31', 'DCA-0,62', 'DCA-1,25','DCA-2,50', 'DCA-5,00', 'DCA-ctrl']


def analyse_silcs(folder):
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.silc')]

    for file in files:
        im = np.load(file).astype('uint8').squeeze()
        im = cv.cvtColor(im, cv.COLOR_BAYER_RG2RGB)  # cv2.COLOR_BAYER_BG2RGB = 48
        cv.imshow('Preview', im)
        cv.waitKey(15)
        # cv.destroyAllWindows()


def main():
    for t in treatment_folders:
        folder = os.path.join(root_folder, date_folder, t)
        analyse_silcs(folder)


main()
