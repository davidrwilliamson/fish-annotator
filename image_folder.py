import numpy as np
import os
import re
import qimage2ndarray as q2n
from cv2 import cvtColor
from PyQt5.QtGui import QPixmap
from typing import TextIO, Tuple


class ImageFolder:
    def __init__(self, folder: str) -> None:
        self.folder = folder

        self._all_files = sorted([file for file in os.listdir(self.folder) if os.path.splitext(file)[1] == '.silc'])
        # self._im_files = []

        self._interesting_frames = []
        self._bad_frames = []
        self._rois = [None] * len(self._all_files)

        self._num_frames = len(self._all_files)

        self._list_image_files()
        # self._list_rois()

        if self._interesting_frames:
            self._curr_frame_no = self._interesting_frames[0]
            self._show_all = False
            self._show_interesting = True
        else:
            self._curr_frame_no = 0
            self._show_all = True
            self._show_interesting = False

    def _load_rois_file(self) -> TextIO:
        rois_filename = os.path.join(self.folder, 'analysis/RoIs')

        if os.path.isfile(rois_filename):
            rois_file = open(rois_filename, 'rt')
        else:
            raise RuntimeError('{}: RoIs file missing.'.format(self.folder))

        return rois_file

    def _list_image_files(self) -> None:
        rois_file = self._load_rois_file()

        # im_files = []
        for line in rois_file.readlines():
            f = re.match('^filename:', line)
            if f:
                fn = line.split(': ')[1].strip()
                self._interesting_frames.append(self._all_files.index(fn))
                # im_files.append(fn)

        # self._im_files = im_files

    # def _list_rois(self) -> None:
    #     rois_file = self._load_rois_file()
    #
    #     roi = []
    #     for line in rois_file.readlines():
    #         r = re.match('^roi:', line)
    #         f = re.match('^filename:', line)
    #         if r:
    #             roi.append(line.split(': ')[1].strip())
    #         elif f:
    #             fn = line.split(': ')[1].strip()
    #             if roi:
    #                 self._rois[self._all_files.index(fn)] = roi
    #                 roi = []
    #         else:
    #             print('Unexpected line in RoIs file.')
    #     self._rois[self._all_files.index(fn)] = roi

    @property
    def curr_files(self) -> Tuple[str, str, str]:
        im_raw = os.path.join(self.folder, self._all_files[self._curr_frame_no])
        im_bg = os.path.join(self.folder, 'analysis/backgrounds',
                             '{}.bg.npy'.format(self._all_files[self._curr_frame_no]))
        im_bm = os.path.join(self.folder, 'analysis/binary_masks',
                             '{}.mask.npy'.format(self._all_files[self._curr_frame_no]))
        return im_raw, im_bg, im_bm

    @property
    def curr_frames(self) -> Tuple[QPixmap, QPixmap, QPixmap, QPixmap]:
        ims = self.curr_files
        im_raw = load_image(ims[0], 'raw')
        if self._curr_frame_no in self._interesting_frames:
            im_bg = load_image(ims[1], 'bg')
            im_bm = load_image(ims[2], 'bm')
            bg_sub = bg_subtract(ims[0], ims[1])
        else:
            im_bg = None
            im_bm = None
            bg_sub = None

        return im_raw, im_bg, im_bm, bg_sub

    @property
    def framepos(self) -> Tuple[int, str]:
        cf_no = self._curr_frame_no
        cf_fn = self._all_files[self._curr_frame_no]

        return cf_no, cf_fn

    @property
    def rois(self) -> list:
        return self._rois[self._curr_frame_no]

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def bad_frames(self) -> list:
        return self._bad_frames

    def go_to_frame(self, frame: int):
        if (frame >= 0) and (frame <= self.num_frames):
            self._curr_frame_no = frame
        else:
            raise

    def next_frame(self) -> None:
        if self._show_all:
            self.go_to_frame((self._curr_frame_no + 1) % self.num_frames)
        elif self._show_interesting:
            int_idx = self._interesting_frames.index(self._curr_frame_no)
            int_idx = (int_idx + 1) % len(self._interesting_frames)
            self.go_to_frame(self._interesting_frames[int_idx])

    def prev_frame(self) -> None:
        if self._show_all:
            self.go_to_frame((self._curr_frame_no - 1) % self.num_frames)
        elif self._show_interesting:
            int_idx = self._interesting_frames.index(self._curr_frame_no)
            int_idx = (int_idx - 1) % len(self._interesting_frames)
            self.go_to_frame(self._interesting_frames[int_idx])

    def toggle_bad_frame(self, checked: bool):
        if checked:
            if self._curr_frame_no not in self._bad_frames:
                self._bad_frames.append(self._curr_frame_no)
        else:
            if self._curr_frame_no in self._bad_frames:
                self._bad_frames.remove(self._curr_frame_no)

    def toggle_interesting_frame(self, checked: bool):
        if checked:
            if self._curr_frame_no not in self._interesting_frames:
                self._interesting_frames.append(self._curr_frame_no)
        else:
            if self._curr_frame_no in self._interesting_frames:
                # This removes the current frame and points us at the "next" one in the list after deletion
                curr_idx = self._interesting_frames.index(self._curr_frame_no)
                next_idx = curr_idx % (len(self._interesting_frames) - 1)
                self._interesting_frames.remove(self._curr_frame_no)
                self.go_to_frame(self._interesting_frames[next_idx])


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
