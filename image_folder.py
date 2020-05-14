import numpy as np
import os
import re
import qimage2ndarray as q2n
from cv2 import cvtColor
from PyQt5.QtGui import QPixmap
from typing import TextIO


class ImageFolder:
    def __init__(self, folder: str) -> None:
        self.folder = folder

        self._all_files = []
        self._im_files = []
        self._bg_files = []
        self._bm_files = []
        self._rois = []

        self._curr_frame_no = 0
        self._no_of_frames = 0

        self._list_image_files()
        self._list_rois()

    def _load_rois_file(self) -> TextIO:
        rois_filename = os.path.join(self.folder, 'analysis/RoIs')

        if os.path.isfile(rois_filename):
            rois_file = open(rois_filename, 'rt')
        else:
            print('{}: RoIs file missing.'.format(self.folder))
            return

        return rois_file

    def _list_image_files(self) -> None:
        rois_file = self._load_rois_file()

        self._all_files = [file for file in os.listdir(self.folder) if os.path.splitext(file)[1] == '.silc']
        im_files = []
        for line in rois_file.readlines():
            f = re.match('^filename:', line)
            if f:
                fn = line.split(': ')[1].strip()
                # fn = os.path.join(self.folder, fn)
                im_files.append(fn)

        self._im_files = im_files
        self._no_of_frames = len(im_files)
        self._bg_files = [file for file in os.listdir(os.path.join(self.folder, 'analysis/backgrounds/'))]
        self._bm_files = [file for file in os.listdir(os.path.join(self.folder, 'analysis/binary_masks/'))]

    def _list_rois(self) -> None:
        rois_file = self._load_rois_file()

        roi = []
        for line in rois_file.readlines():
            r = re.match('^roi:', line)
            f = re.match('^filename:', line)
            if r:
                roi.append(line.split(': ')[1].strip())
            elif f:
                if roi:
                    self._rois.append(roi)
                    roi = []
            else:
                print('Unexpected line in RoIs file.')
        self._rois.append(roi)

    @property
    def num_images(self) -> int:
        return len(self._im_files)

    @property
    def curr_files(self) -> [str, str, str]:
        im_raw = os.path.join(self.folder, self._im_files[self._curr_frame_no])
        im_bg = os.path.join(self.folder, 'analysis/backgrounds',
                             '{}.bg.npy'.format(self._im_files[self._curr_frame_no]))
        im_bm = os.path.join(self.folder, 'analysis/binary_masks',
                             '{}.mask.npy'.format(self._im_files[self._curr_frame_no]))
        return im_raw, im_bg, im_bm

    @property
    def curr_frames(self) -> [QPixmap, QPixmap, QPixmap, QPixmap]:
        ims = self.curr_files
        im_raw = load_image(ims[0], 'raw')
        im_bg = load_image(ims[1], 'bg')
        im_bm = load_image(ims[2], 'bm')
        bg_sub = bg_subtract(ims[0], ims[1])

        return im_raw, im_bg, im_bm, bg_sub

    @property
    def framepos(self) -> [int, str]:
        cf_no = self._curr_frame_no
        cf_fn = self._im_files[self._curr_frame_no]

        return cf_no, cf_fn

    @property
    def rois(self) -> list:
        return self._rois[self._curr_frame_no]

    @property
    def num_frames(self) -> int:
        return self._no_of_frames

    def next_image(self) -> None:
        self._curr_frame_no = (self._curr_frame_no + 1) % self.num_images

    def prev_image(self) -> None:
        self._curr_frame_no = (self._curr_frame_no - 1) % self.num_images


def load_image(file: str, im_type: str = 'raw') -> QPixmap:
    if im_type == 'bm':
        im = np.load(file).astype('uint8').squeeze() * 255
    else:
        im = np.load(file).astype('uint8').squeeze()
        im = cvtColor(im, 48)  # cv2.COLOR_BAYER_BG2RGB = 48
    im = q2n.array2qimage(im)
    im = QPixmap.fromImage(im)

    return im


def bg_subtract(im_raw: str, im_bg: str) -> QPixmap:
    im_raw, im_bg = np.load(im_raw).squeeze(), np.load(im_bg).squeeze()

    im = im_raw - im_bg
    im += 215
    im[im < 0] = 0
    im[im > 255] = 255
    im = np.uint8(im)

    im = cvtColor(im, 48)
    im = q2n.array2qimage(im)
    im = QPixmap.fromImage(im)

    return im
