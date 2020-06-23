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
        self._interesting_frames = []
        self._bad_frames = []
        self._rois = [None] * len(self._all_files)

        self._num_frames = len(self._all_files)

        self._list_image_files()
        self._list_rois()

        self._intf_idx = None

        if self._interesting_frames:
            self._intf_idx = 0
            self._curr_frame_no = self._interesting_frames[self._intf_idx]
            # self._show_bad = False
            self._show_interesting = True
            self._show_other = False
        else:
            self._curr_frame_no = 0
            # self._show_bad = False
            self._show_interesting = False
            self._show_other = True

    def _load_rois_file(self) -> TextIO:
        rois_filename = os.path.join(self.folder, 'analysis/RoIs')

        if os.path.isfile(rois_filename):
            rois_file = open(rois_filename, 'rt')
        else:
            raise RuntimeError('{}: RoIs file missing.'.format(self.folder))
        # TODO: Handle this gracefully in program rather than crashing out

        return rois_file

    def _list_image_files(self) -> None:
        rois_file = self._load_rois_file()

        for line in rois_file.readlines():
            f = re.match('^filename:', line)
            if f:
                fn = line.split(': ')[1].strip()
                self._interesting_frames.append(self._all_files.index(fn))

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
                    self._rois[self._all_files.index(fn)] = roi
                    roi = []
                fn = line.split(': ')[1].strip()
            else:
                print('Unexpected line in RoIs file.')
        self._rois[self._all_files.index(fn)] = roi

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

        if os.path.isfile(ims[1]):
            im_bg = load_image(ims[1], 'bg')
            bg_sub = bg_subtract(ims[0], ims[1])
        else:
            im_bg = None
            bg_sub = None
        if os.path.isfile(ims[2]):
            im_bm = load_image(ims[2], 'bm')
        else:
            im_bm = None

        return im_raw, im_bg, im_bm, bg_sub

    @property
    def framepos(self) -> Tuple[int, str]:
        cf_no = self._curr_frame_no
        cf_fn = self._all_files[self._curr_frame_no]

        return cf_no, cf_fn

    @property
    def frames(self):
        # Generator that yields every frame in the current scope, in order
        curr_frame = self._curr_frame_no
        self.go_to_first_frame()
        while self._curr_frame_no < self.last_frame:
            yield curr_frame
            self.next_frame()
        yield curr_frame
        # This returns us to the frame we were at before frames generator was called
        self.go_to_frame(curr_frame)

    @property
    def rois(self) -> list:
        return self._rois[self._curr_frame_no]

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def last_frame(self) -> int:
        # This returns the INDEX of the last frame under current show other/interesting rules, not the frame itself
        if self._show_other and self._show_interesting:
            return self._num_frames - 1
        elif self._show_other and not self._show_interesting:
            for i in range(self._num_frames - 1, 0, -1):
                if i not in self._interesting_frames:
                    return i
        elif self._show_interesting and not self._show_other:
            return self._interesting_frames[-1]

    def go_to_frame(self, frame: int) -> None:
        if (frame >= 0) and (frame <= self.num_frames):
            self._curr_frame_no = frame
            if self._curr_frame_no in self._interesting_frames:
                self._intf_idx = self._interesting_frames.index(self._curr_frame_no)
        else:
            raise

    def go_to_first_frame(self) -> None:
        if self._show_other and self._show_interesting:
            self.go_to_frame(0)
        elif self._show_other and not self._show_interesting:
            self.go_to_frame(0)
            if self._curr_frame_no in self._interesting_frames:
                self.next_frame()
        elif self._show_interesting and not self._show_other:
            self._intf_idx = 0
            self.go_to_frame(self._interesting_frames[self._intf_idx])

    def next_frame(self) -> None:
        if self._show_other and self._show_interesting:
            self.go_to_frame((self._curr_frame_no + 1) % self.num_frames)
        elif self._show_other and not self._show_interesting:
            self.go_to_frame((self._curr_frame_no + 1) % self.num_frames)
            if self._curr_frame_no in self._interesting_frames:
                self.next_frame()
        elif self._show_interesting and not self._show_other:
            # There is a bug were if multiple frames are unchecked around start/end frame we get list index out of range
            # TODO: Pin down and fix this bug
            if self._curr_frame_no >= self._interesting_frames[self._intf_idx]:
                self._intf_idx = (self._intf_idx + 1) % len(self._interesting_frames)
            self.go_to_frame(self._interesting_frames[self._intf_idx])
        else:  # If neither show other or show interesting are checked, do nothing
            pass

    def prev_frame(self) -> None:
        if self._show_other and self._show_interesting:
            self.go_to_frame((self._curr_frame_no - 1) % self.num_frames)
        elif self._show_other and not self._show_interesting:
            self.go_to_frame((self._curr_frame_no - 1) % self.num_frames)
            if self._curr_frame_no in self._interesting_frames:
                self.prev_frame()
        elif self._show_interesting and not self._show_other:
            if self._curr_frame_no <= self._interesting_frames[self._intf_idx]:
                self._intf_idx = (self._intf_idx - 1) % len(self._interesting_frames)
            self.go_to_frame(self._interesting_frames[self._intf_idx])
        else:  # If neither show other or show interesting are checked, do nothing
            pass

    def toggle_bad_frame(self, checked: bool) -> None:
        if checked:
            if self._curr_frame_no not in self._bad_frames:
                self._bad_frames.append(self._curr_frame_no)
                self._bad_frames.sort()
        else:
            if self._curr_frame_no in self._bad_frames:
                self._bad_frames.remove(self._curr_frame_no)

    def toggle_interesting_frame(self, checked: bool) -> None:
        if checked:
            if self._curr_frame_no not in self._interesting_frames:
                self._interesting_frames.append(self._curr_frame_no)
                self._interesting_frames.sort()
        else:
            if self._curr_frame_no in self._interesting_frames:
                self._interesting_frames.remove(self._curr_frame_no)
                if not self._show_other:
                    self.next_frame()


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
