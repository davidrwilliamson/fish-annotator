import cv2 as cv
import numpy as np
import os
import pysilcam.background as scbg
import pysilcam.process as scpr
import skimage as sk
import re
from tqdm import tqdm

import matplotlib.patches as mpatches


def find_circles(im_file):
    im = np.load(im_file).astype('uint8').squeeze()
    im = cv.cvtColor(im, cv.COLOR_BAYER_BG2RGB)
    im = cv.medianBlur(im, 7)
    g_im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(g_im, cv.HOUGH_GRADIENT, 1, minDist=200,
                              param1=70, param2=50, minRadius=100, maxRadius=300)

    return circles


class BackgroundSubtraction:

    def __init__(self, folder='/home/davidw/'):
        self.av_window = 30
        self.folder = folder
        self.files = ''
        self.total_files = -1
        self.aqgen = None
        self.bggen = None

    def initialise(self):
        if self.files == '':
            self.files = [os.path.join(f)
                          for f in sorted(os.listdir(self.folder))
                          if f.endswith('.silc_bayer') or f.endswith('.bayer_silc') or f.endswith('.silc')]

        self.total_files = len(self.files)

        self.aqgen = self._generator_files()

        print('* Initializing background image handler')
        self.bggen = self._bayer_backgrounder(self.av_window, self.aqgen, None, True)

    def calc_backgrounds(self, interesting_only=True):
        # Basically reimplements the silcam background subtracter but can also "look forward" for the first n files w/ n < averaging window
        interest_files = []

        if interesting_only:
            rois_filename = os.path.join(self.folder, 'analysis/RoIs')
            if os.path.isfile(rois_filename):
                rois_file = open(rois_filename, 'rt')
                for line in rois_file.readlines():
                    f = re.match('^filename:', line)
                    if f:
                        interest_files.append(line.split(': ')[1].strip())
        else:
            print('RoIs file missing')
            interest_files = self.files

        print('Calculating backgrounds...')
        for file in tqdm(interest_files):
            # Skip if a background file already exists
            if os.path.isfile('{}.bg.npy'.format(os.path.join(self.folder, 'analysis/backgrounds', file))):
                pass
            else:
                fn = os.path.join(self.folder, file)
                im0 = np.load(fn).squeeze()
                try:
                    f_index = self.files.index(file)
                    bg_stack = np.empty((self.av_window, im0.shape[0], im0.shape[1]))
                    # Files greater than the rolling average window
                    if f_index >= self.av_window:
                        for i in range(f_index - self.av_window, f_index):
                            bg_stack[i - f_index] = np.load(os.path.join(self.folder, self.files[i])).squeeze()
                    # Earlier files
                    else:
                        for i in range(self.av_window):
                            bg_stack[i] = np.load(os.path.join(self.folder, self.files[i])).squeeze()
                    bg = np.uint8(bg_stack.mean(axis=0))
                    np.save('{}.bg.npy'.format(os.path.join(self.folder, 'analysis/backgrounds', file)), bg.squeeze())
                except FileNotFoundError:
                    print('Filename {} not found in list of .silc files.'.format(file))
        print('Background calculation complete.')

    def calc_rois(self):
        out_filename = os.path.join(self.folder, 'analysis', 'RoIs')
        if os.path.isfile(out_filename):
            print('RoI file already exists for {}/'.format(self.folder))
        out_file = open(out_filename, 'wt')

        print ('Performing background subtraction and RoI extraction...')

        for file in tqdm(self.files):
            try:
                im_raw = np.load(os.path.join(self.folder, file)).squeeze()
                im_bg = np.load(os.path.join(self.folder, 'analysis/backgrounds/', '{}.bg.npy'.format(file))).squeeze()
                # Background subtraction
                im = im_raw - im_bg
                im += 215
                im[im < 0] = 0
                im[im > 255] = 255
                im = np.uint8(im)

                im_bm, rects = mask(im, 0.9)

                if len(rects) > 0:
                    np.save('{}.mask.npy'.format(os.path.join(self.folder, 'analysis/binary_masks', file)), im_bm)
                    out_file.write('filename: {}\n'.format(file),)
                    out_file.writelines(
                        ['roi: {0}, {1}, {2}, {3}\n'.format(rect._x0, rect._y0, rect._x1, rect._y1) for rect in rects])
            except FileNotFoundError:
                print('Filename {} not found in list of .silc files.'.format(file))

        out_file.close()
        print('Complete.')

    def _bayer_backgrounder(self, av_window, acquire, bad_lighting_limit=None, real_time_stats=False):
        '''Similar to backgrounder in the main pysilcam code but adjusted for Bayer8'''
        bgstack, imbg = scbg.ini_background(av_window, acquire)
        stacklength = len(bgstack)

        for filename, imraw in acquire:
            bgstack, imbg, imc = scbg.shift_and_correct(bgstack, imbg, imraw,
                                                        stacklength, real_time_stats)

            yield filename, imc, imraw

    def _generator_files(self):
        '''Generator that loads images from file and converts them to RGB'''
        for f in tqdm(self.files):
            try:
                im = np.load(f)
                im = cv.cvtColor(im, cv.COLOR_BAYER_BG2RGB)
            except EOFError:
                print ('Error loading file {}'.format(repr(f)))
                continue

            yield f, im


def background_subtraction(folder):
    bs = BackgroundSubtraction(folder)
    bs.initialise()
    bs.calc_backgrounds(False)
    bs.calc_rois()


def mask(im, thresh, min_area=1000):
    # Convert to binary image, selecting all pixels with at least one channel > 0 and setting those pixels to 1
    im_bw = scpr.image2binary_fast(im, thresh)
    # im_bin = scpr.clean_bw(im_bin, 12)
    # This was needed for RGB images (?) but not when operating on single channel Bayers
    # im_bw = np.amax(im_bw, axis=2)
    im_bw = im_bw * np.ones(im_bw.shape)

    # Label regions, discard ones that are too small, return bounding boxes of the rest
    im_lab = sk.measure.label(im_bw)
    props = sk.measure.regionprops(im_lab)
    rects = []
    for i, el in enumerate(props):
        if el.area < min_area:
            for c in el.coords:
                im_bw[tuple(c)] = 0.
        else:
            minr, minc, maxr, maxc = el.bbox
            if (minc > 0) and (maxc < im_bw.shape[1]) and (minr > 0) and (maxr < im_bw.shape[0]): # Discard RoIs that extend to the L/R edges of the image
                rects.append(mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))

    mask = np.array(im_bw, dtype='bool')

    return mask, rects