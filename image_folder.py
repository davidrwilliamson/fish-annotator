import numpy as np
import os
import re
import qimage2ndarray as q2n
from random import randrange, choice
from cv2 import cvtColor
from PyQt5.QtGui import QPixmap
from typing import TextIO, Tuple
import errors


class ImageFolder:
    def __init__(self, folder: str) -> None:
        self.folder = folder

        silc_extensions = ['.silc', '.silc_bayer', '.bayer_silc']

        self._all_files = sorted([file for file in os.listdir(self.folder) if os.path.splitext(file)[1] in silc_extensions])
        self._check_image_files()

        self._interesting_frames = []
        self._bad_frames = []
        self._rois = [None] * len(self._all_files)

        self._num_frames = len(self._all_files)

        self._list_image_files()
        self._list_rois()

        self._intf_idx = 0
        self._bad_idx = 0
        self.frame_w, self.frame_h = self._get_frame_size()

        if self._interesting_frames:
            self._intf_idx = 0
            self._curr_frame_no = self._interesting_frames[self._intf_idx]
            self._show_bad = False
            self._show_interesting = True
            self._show_other = False
        else:
            self._curr_frame_no = 0
            self._show_bad = False
            self._show_interesting = False
            self._show_other = True

    def _check_image_files(self) -> None:
        if len(self._all_files) == 0:
            raise errors.NoImageFilesError(self.folder)

    def _get_frame_size(self) -> Tuple[int, int]:
        im0 = np.load(os.path.join(self.folder, self._all_files[0])).astype('uint8').squeeze()
        h, w = im0.shape

        return w, h

    def _load_rois_file(self) -> TextIO:
        rois_filename = os.path.join(self.folder, 'analysis/RoIs')

        if os.path.isfile(rois_filename) and os.stat(rois_filename).st_size > 0:
            rois_file = open(rois_filename, 'rt')
        else:
            rois_file = None
            # raise RuntimeError('{}: RoIs file missing.'.format(self.folder))
        # TODO: Handle this gracefully in program rather than crashing out

        return rois_file

    def _load_bad_frames_file(self) -> TextIO:
        bad_frames_filename = os.path.join(self.folder, 'analysis/bad_frames')

        if os.path.isfile(bad_frames_filename) and os.stat(bad_frames_filename).st_size > 0:
            bad_frames_file = open(bad_frames_filename, 'rt')
        else:
            bad_frames_file = None
            # raise RuntimeError('{}: RoIs file missing.'.format(self.folder))
        # TODO: Handle this gracefully in program rather than crashing out

        return bad_frames_file

    def _list_image_files(self) -> None:
        rois_file = self._load_rois_file()
        bad_frames_file = self._load_bad_frames_file()

        if rois_file:
            for line in rois_file.readlines():
                f = re.match('^filename:', line)
                if f:
                    fn = line.split(': ')[1].strip()
                    self._interesting_frames.append(self._all_files.index(fn))
        self._interesting_frames.sort()

        if bad_frames_file:
            for line in bad_frames_file.readlines():
                f = re.match('^filename:', line)
                if f:
                    fn = line.split(': ')[1].strip()
                    self._bad_frames.append(self._all_files.index(fn))
        self._bad_frames.sort()

    def _list_rois(self) -> None:
        rois_file = self._load_rois_file()

        if rois_file:
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
            if self._curr_frame_no in self._bad_frames:
                self._bad_idx = self._bad_frames.index(self._curr_frame_no)
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

    def random_frame(self) -> None:
        # Go to a random frame in the current context, useful for sampling for annotation
        if self._show_other and self._show_interesting:
            f = randrange(self._num_frames)
        elif self._show_other and not self._show_interesting:
            f = randrange(self._num_frames)
            if f in self._interesting_frames or f in self._bad_frames:
                self.random_frame()
        elif self._show_interesting and not self._show_other:
            f0 = randrange(len(self._interesting_frames))
            f = self._interesting_frames[f0]
        self.go_to_frame(f)

    def toggle_bad_frame(self, checked: bool) -> None:
        if checked:
            if self._curr_frame_no not in self._bad_frames:
                self._add_bad_frame()
        else:
            if self._curr_frame_no in self._bad_frames:
                self._remove_bad_frame()

    def toggle_interesting_frame(self, checked: bool) -> None:
        # TODO: Removing the last interesting frame when only interesting frames is checked or unchecking other frames
        # crashes the program. In cases where there are no interesting frames we need to fall back to showing other
        # Same for bad frames.
        if checked:
            if self._curr_frame_no not in self._interesting_frames:
                self._interesting_frames.append(self._curr_frame_no)
                self._interesting_frames.sort()
                # Calculate RoIs (if any), add file to RoIs list file and write binary mask to disk
                self.calc_single_roi()
        else:
            if self._curr_frame_no in self._interesting_frames:
                # Rewrite the RoIs file without this frame
                self.remove_roi()
                # If this is the last interesting frame, go back one so we don't get out of range
                if self._curr_frame_no is self._interesting_frames[-1]:
                    self._intf_idx -= 1
                self._interesting_frames.remove(self._curr_frame_no)
                if not self._show_other:
                    self.next_frame()

    def toggle_show_bad(self, checked: bool) -> None:
        self._show_bad = checked
        self.next_frame()
        self.prev_frame()

    def toggle_show_interesting(self, checked: bool) -> None:
        self._show_interesting = checked
        self.next_frame()
        self.prev_frame()

    def toggle_show_other(self, checked: bool) -> None:
        self._show_other = checked
        self.next_frame()
        self.prev_frame()

    def calc_single_roi(self) -> None:
        """Calculate a binary mask and RoIs for a single file, to be used when marking images as 'interesting'.
        Note that we write out the BM and add the file to the RoIs list even if no areas of interest are found."""
        from analysis import mask

        im_raw, im_bg, _ = self.curr_files  # returns str, str, str
        im_raw = np.load(im_raw).squeeze()
        im_bg = np.load(im_bg).squeeze()

        im = im_raw - im_bg
        im += 215
        im[im < 0] = 0
        im[im > 255] = 255
        im = np.uint8(im)
        im_bm, rects = mask(im, 0.9)

        fname = self._all_files[self._curr_frame_no]

        rois_filename = os.path.join(self.folder, 'analysis', 'RoIs')
        if os.path.isfile(rois_filename):
            print('RoI file already exists for {}/'.format(self.folder))
            rois_file = open(rois_filename, 'at')
        else:
            rois_file = open(rois_filename, 'wt')

        np.save('{}.mask.npy'.format(os.path.join(self.folder, 'analysis/binary_masks', fname)), im_bm)
        rois_file.write('filename: {}\n'.format(fname), )
        if len(rects) > 0:
            rois_file.writelines(
                ['roi: {0}, {1}, {2}, {3}\n'.format(rect._x0, rect._y0, rect._x1, rect._y1) for rect in rects])

    def remove_roi(self) -> None:
        rois_filename = os.path.join(self.folder, 'analysis', 'RoIs')
        temp_filename = os.path.join(self.folder, 'analysis', 'temp')

        # Extract line numbers to be removed
        # We need to find both the line specifying the current file, and any RoIs lines following it
        with open(rois_filename, 'rt') as rois:
            counter = 0
            line_numbers = []
            line_found = False

            lines = rois.readlines()
            for line in lines:
                line = line.strip()
                if line == 'filename: {}'.format(os.path.basename(self.curr_files[0])):
                    line_numbers.append(counter)
                    line_found = True
                    # Remove associated binary mask file
                    # im_bm = self.curr_files[2]
                    # os.remove(im_bm)
                elif line_found:
                    if re.match('^roi:.*', line):
                        line_numbers.append(counter)
                    else:
                        line_found = False

                counter += 1

        # Rewrite file without offending lines
        with open(rois_filename, 'rt') as rois, open(temp_filename, 'wt') as temp:
            counter = 0
            lines = rois.readlines()
            for line in lines:
                if counter not in line_numbers:
                    temp.write(line)
                counter += 1

        # Clean up
        rois.close()
        temp.close()
        os.remove(rois_filename)
        os.rename(temp_filename, rois_filename)

    def _add_bad_frame(self) -> None:
        bad_frames_filename = os.path.join(self.folder, 'analysis', 'bad_frames')
        fname = os.path.basename(self.curr_files[0])

        if os.path.isfile(bad_frames_filename):
            with open(bad_frames_filename, 'at') as f:
                f.write('filename: {}\n'.format(fname))
        else:
            with open(bad_frames_filename, 'wt') as f:
                f.write('filename: {}\n'.format(fname))

        self._bad_frames.append(self._curr_frame_no)
        self._bad_frames.sort()

    def _remove_bad_frame(self) -> None:
        self._bad_frames.remove(self._curr_frame_no)

        # TODO: Remove bad frame line from bad frames file


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
