import numpy as np
import pandas as pd
import os
import cv2
import pysilcam.background as scbg
import pysilcam.process as scpr
import skimage as sk
import re

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

# Test background subtraction


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
            self.files = [os.path.join(self.folder, f)
                          for f in sorted(os.listdir(self.folder)) if f.endswith('.silc_bayer')]

        self.total_files = len(self.files)

        self.aqgen = self.generator_files()

        print('* Initializing background image handler')
        self.bggen = self.bayer_backgrounder(self.av_window, self.aqgen, None, True)

    def bayer_backgrounder(self, av_window, acquire, bad_lighting_limit=None, real_time_stats=False):
        '''Similar to backgrounder in the main pysilcam code but adjusted for Bayer8'''
        bgstack, imbg = scbg.ini_background(av_window, acquire)
        stacklength = len(bgstack)

        for filename, imraw in acquire:
            bgstack, imbg, imc = scbg.shift_and_correct(bgstack, imbg, imraw,
                                                        stacklength, real_time_stats)

            yield filename, imc, imraw

    def generator_files(self):
        '''Generator that loads images from file and converts them to RGB'''
        for f in tqdm(self.files):
            try:
                im = np.load(f)
                im = cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB)
            except EOFError:
                print ('Error loading file {}'.format(repr(f)))
                continue

            yield f, im


def mask(im, thresh, min_area=10000):
    # Convert to binary image, selecting all pixels with at least one channel > 0 and setting those pixels to 1
    im_bw = scpr.image2binary_fast(im, thresh)
    # im_bin = scpr.clean_bw(im_bin, 12)
    im_bw = np.amax(im_bw, axis=2)
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

    mask = np.array(im_bw, dtype='uint8')

    return mask, rects


def save_rois(bs):
    last_frame = bs.total_files
    out_filename = os.path.join(bs.folder, 'RoIs')
    if os.path.isfile(out_filename):
        print ('RoI file already exists for {}/'.format(bs.folder))
        return
    else:
        out_file = open(out_filename, 'wt')
        print ('Extracting RoIs from {}/'.format(bs.folder))

    try:
        for i in range(last_frame):  # bs.total_files:
            filename, imc, imraw = next(bs.bggen)
            im_bw, rects = mask(imc, 0.9)
            short_fn = os.path.basename(filename)

            if len(rects) > 0:
                np.save('{}.mask'.format(filename), im_bw)
                out_file.write('filename: {}\n'.format(short_fn))
                out_file.writelines(['roi: {0}, {1}, {2}, {3}\n'.format(rect._x0, rect._y0, rect._x1 ,rect._y1) for rect in rects])
    except StopIteration:
        print ('RoI extraction complete.')
    out_file.close()


def display_rois(folder):
    rois_filename = os.path.join(folder, 'RoIs')
    if os.path.isfile(rois_filename):
        rois_file = open(rois_filename, 'rt')
    else:
        print ('RoIs file missing')
        return

    files = []
    rois = []
    all_rois = []
    for line in rois_file.readlines():
        f = re.match('^filename:', line)
        if f:
            if len(rois) > 0:
                all_rois.append(rois)
            files.append(line.split(': ')[1].strip())
            rois = []
        else:
            r = re.match('^roi:', line)
            if r:
                rois.append(line.split(': ')[1].strip())
            else:
                print ("Unexpected line")
    if len(rois) > 0:
        all_rois.append(rois)

    fig = plt.figure(0)
    ax = plt.gca()

    def update(frame):
        fn = os.path.join(folder, files[frame])
        im = np.load(fn)
        im = np.array(cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB))
        # im_bw = np.load('{}.mask.npy'.format(fn))
        # im = np.array(cv2.cvtColor(im, cv2.COLOR_BAYER_BG2RGB) * np.stack((im_bw,) * 3, axis=-1), dtype='uint8')

        timestamp = pd.to_datetime(os.path.splitext(os.path.split(fn)[-1])[0][1:])
        fig.canvas.set_window_title('Timestamp: {0}, {1}/{2}'.format(timestamp, frame, len(all_rois)))

        plt.cla()
        ax.imshow(im)
        for r in all_rois[frame]:
            bbox = np.fromstring(r, dtype='int', sep=',')
            rect = mpatches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)
            plt.annotate('{} px'.format(rect.get_width()),
                         (bbox[0] + (rect.get_width() / 2), bbox[1]),
                         textcoords='offset points',
                         xytext=(0, 7),
                         ha='center',
                         color='red',)
            plt.annotate('{} px'.format(rect.get_height()),
                         (bbox[0], bbox[1] + (rect.get_height() / 2)),
                         textcoords='offset points',
                         xytext=(-7, 5),
                         ha='center',
                         rotation=90,
                         color='red')

    animation = FuncAnimation(fig, update, frames=len(files), interval=50, save_count=0)
    plt.show()
    plt.close(fig)


def output_backgrounds(folder):
    stack_size = 50
    all_files = [f for f in os.listdir(folder) if f.endswith('.silc_bayer')]

    rois_filename = os.path.join(folder, 'RoIs')
    if os.path.isfile(rois_filename):
        rois_file = open(rois_filename, 'rt')
    else:
        print('RoIs file missing')
        return

    interest_files = []
    for line in rois_file.readlines():
        f = re.match('^filename:', line)
        if f:
            interest_files.append(line.split(': ')[1].strip())

    for file in interest_files:
        fn = os.path.join(folder, file)
        im0 = np.load(fn).squeeze()
        try:
            f_index = all_files.index(file)
            bg_stack = np.empty((stack_size, im0.shape[0], im0.shape[1]))
            if f_index >= stack_size:
                for i in range(f_index - stack_size, f_index):
                    bg_stack[i - f_index] = np.load(os.path.join(folder, all_files[i])).squeeze()
            else:
                for i in range(stack_size):
                    bg_stack[i] = np.load(os.path.join(folder, all_files[i])).squeeze()
            bg = bg_stack.mean(axis=0)
            np.save('{}.bg'.format(os.path.join(folder, file)), bg)
        except ValueError:
            print('Filename {} not found in list of .silc files.'.format(file))
    print ('Completed {}'.format(folder))


def play_animation(bs):
    # Animated frame by frame output display
    fig, axarr = plt.subplots(3, 1)

    def update(frame):
        filename, imc, imraw = next(bs.bggen)
        im_bw, rects = mask(imc, 0.9)

        timestamp = pd.to_datetime(os.path.splitext(os.path.split(filename)[-1])[0][1:])
        fig.canvas.set_window_title('Timestamp: {0}, {1}/{2}'.format(timestamp, frame, bs.total_files))
        plt.cla()
        axarr[0].imshow(imraw)
        plt.cla()
        axarr[1].imshow(imc)
        plt.cla()
        axarr[2].imshow(im_bw)
        for rect in rects:
            axarr[2].add_patch(rect)
        return axarr

    animation = FuncAnimation(fig, update, frames=bs.total_files, interval=1, save_count=0)
    plt.show()
    plt.close(fig)


def main():
    folders = ['/media/dave/dave_8tb/2021/20210422/sw1_1'
        # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200404/3',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200405',
                # OSError: Failed to interpret file '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200406/1/D20200406T120745.162977.silc' as a pickle
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200406/1',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200407/1',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200408/1',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200409/1',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200410/1',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200411/1',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200412/2',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200413/1',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200414/1',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200415/1',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200416/1',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200417/1',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200408/DCA-0,15',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200412/DCA-5,00',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200408/DCA-0,3   1',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200412/DCA-ctrl-2',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200408/DCA-0,62',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200413/DCA-0,15',
                # # error with this one during saverois on second-last frame. worked fine after rerunning though
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200408/DCA-1,25',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200413/DCA-0,31',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200408/DCA-2,50',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200413/DCA-0,62',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200408/DCA-5,00',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200413/DCA-1,25',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200408/DCA-ctrl',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200413/DCA-2,50',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200409/DCA-0,15',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200413/DCA-ctrl',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200409/DCA-0,31',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200414/DCA-0,15',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200409/DCA-0,62',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200414/DCA-0,31',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200409/DCA-1,25',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200414/DCA-0,62',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200409/DCA-2,50',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200414/DCA-1,25',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200409/DCA-5,00',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200414/DCA-2,50',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200409/DCA-ctrl',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200414/DCA-ctrl',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200410/DCA-0,15',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200415/DCA-0,15',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200410/DCA-0,31',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200415/DCA-0,31',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200410/DCA-0,62',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200415/DCA-0,62',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200410/DCA-1,25',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200415/DCA-1,25',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200410/DCA-2,50',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200415/DCA-2,50',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200410/DCA-5,00',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200415/DCA-ctrl',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200410/DCA-ctrl',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200416/DCA-0,15',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200411/DCA-0,15',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200416/DCA-0,31',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200411/DCA-0,31',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200416/DCA-0,62',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200411/DCA-0,62',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200416/DCA-1,25',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200411/DCA-1,25',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200416/DCA-2,50',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200411/DCA-2,50',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200416/DCA-ctrl',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200411/DCA-5,00',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200417/DCA-0,15',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200411/DCA-ctrl',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200417/DCA-0,31',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200412/DCA-0,15',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200417/DCA-0,62',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200412/DCA-0,31',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200417/DCA-1,25',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200412/DCA-0,62',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200417/DCA-2,50',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200412/DCA-1,25',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200417/DCA-ctrl',
                # '/media/davidw/SINTEF Polar Night D/Easter cod experiments/Bernard/20200412/DCA-2,50'
                ]

    for folder in folders:
        output_backgrounds(folder)
        bs = BackgroundSubtraction(folder)
        bs.initialise()

        # play_animation(bs)
        save_rois(bs)
        display_rois(folder)



main()