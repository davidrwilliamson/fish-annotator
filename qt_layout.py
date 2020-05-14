from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize, Qt, QRect, pyqtSignal, pyqtSlot
from PyQt5.QtGui import *
import numpy as np
import qimage2ndarray as q2n
from cv2 import cvtColor
import os, re


class ImageFolder:
    def __init__(self, folder):
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

    def _load_rois_file(self):
        rois_filename = os.path.join(self.folder, 'analysis/RoIs')

        if os.path.isfile(rois_filename):
            rois_file = open(rois_filename, 'rt')
        else:
            print('{}: RoIs file missing.'.format(self.folder))
            return

        return rois_file

    def _list_image_files(self):
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

    def _list_rois(self):
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
    def num_images(self):
        return len(self._im_files)

    @property
    def curr_files(self):
        im_raw = os.path.join(self.folder, self._im_files[self._curr_frame_no])
        im_bg = os.path.join(self.folder, 'analysis/backgrounds', '{}.bg.npy'.format(self._im_files[self._curr_frame_no]))
        im_bm = os.path.join(self.folder, 'analysis/binary_masks', '{}.mask.npy'.format(self._im_files[self._curr_frame_no]))
        return im_raw, im_bg, im_bm

    @property
    def curr_frames(self):
        ims = self.curr_files
        im_raw = load_image(ims[0], 'raw')
        im_bg = load_image(ims[1], 'bg')
        im_bm = load_image(ims[2], 'bm')
        bg_sub = bg_subtract(ims[0], ims[1])
        return im_raw, im_bg, im_bm, bg_sub

    @property
    def framepos(self):
        cf_no = self._curr_frame_no
        cf_fn = self._im_files[self._curr_frame_no]

        return cf_no, cf_fn

    @property
    def rois(self):
        return self._rois[self._curr_frame_no]

    @property
    def num_frames(self):
        return self._no_of_frames

    def next_image(self):
        self._curr_frame_no = (self._curr_frame_no + 1) % self.num_images

    def prev_image(self):
        self._curr_frame_no = (self._curr_frame_no - 1) % self.num_images


def load_image(file, im_type='raw'):
    if im_type == 'bm':
        im = np.load(file).astype('uint8').squeeze() * 255
    else:
        im = np.load(file).astype('uint8').squeeze()
        im = cvtColor(im, 48)  # cv2.COLOR_BAYER_BG2RGB = 48
    im = q2n.array2qimage(im)
    im = QPixmap.fromImage(im)

    return im


def bg_subtract(im_raw, im_bg):
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.im_folder = None
        self.curr_layer = 0
        self.draw_rois = False

        # Menubar
        main_menu = MainMenu(self)
        self.setMenuWidget(main_menu)

        self.image_frame = ImageFrame(self)
        self.paint_canvas = PaintingCanvas(self)
        self.rois_canvas = RoIsCanvas(self)
        self.lbl_frame = QtWidgets.QLabel('\n')
        self.lbl_frame_no = QtWidgets.QLabel('\n')
        self.lbl_frame_no.setAlignment(Qt.AlignRight)

        # Set up layout of window inside central placeholder widget
        placeholder = QtWidgets.QWidget()
        self.setCentralWidget(placeholder)
        grid_layout = QtWidgets.QGridLayout(placeholder)
        grid_layout.addWidget(self.lbl_frame, 0, 0)
        grid_layout.addWidget(self.lbl_frame_no, 0, 0)
        grid_layout.addWidget(self.image_frame, 1, 0)
        grid_layout.addWidget(self.rois_canvas, 1, 0)
        grid_layout.addWidget(self.paint_canvas, 1, 0)

        # Add various sets of buttons
        bottom_buttons = BottomButtons()
        grid_layout.addWidget(bottom_buttons, 2, 0)
        right_buttons = RightButtons()
        grid_layout.addWidget(right_buttons, 1, 1)
        br_buttons = BottomRightButtons()
        grid_layout.addWidget(br_buttons, 2, 1)

        # Connect up button signals
        main_menu.sgnl_im_folder.connect(self.set_im_folder)
        right_buttons.sgnl_change_im_layer.connect(self.change_layer)
        right_buttons.sgnl_toggle_rois.connect(self.toggle_rois)
        bottom_buttons.sgnl_change_frame.connect(self.change_frame)
        br_buttons.sgnl_cb_bad_changed.connect(self.bad_frame)
        br_buttons.sgnl_cb_interest_changed.connect(self.interesting_frame)

        self.setWindowTitle("Fish Annotator")

    def update_lbl_frame(self):
        folder = self.im_folder.folder
        cf_no, cf_fn = self.im_folder.framepos
        num_frames = self.im_folder.num_frames
        self.lbl_frame.setText('Folder:  {}\nFile:      {}'.format(folder, cf_fn))
        self.lbl_frame_no.setText('\nFrame: {} / {}'.format(cf_no, num_frames))

    @pyqtSlot(ImageFolder)
    def set_im_folder(self, im_folder):
        self.im_folder = im_folder
        self.change_layer(0)

    @pyqtSlot(int)
    def change_layer(self, im_idx):
        curr_frames = self.im_folder.curr_frames
        im = curr_frames[im_idx]
        self.curr_layer = im_idx
        self.image_frame.update_image(im)
        self.update_lbl_frame()

    @pyqtSlot(bool)
    def toggle_rois(self, checked):
        self.draw_rois = checked
        if checked:
            rois = self.im_folder.rois
            self.rois_canvas.draw_rois(rois)
        else:
            self.rois_canvas.erase_rois()

    @pyqtSlot(int)
    def change_frame(self, direction):
        if direction > 0:
            self.im_folder.next_image()
        if direction < 0:
            self.im_folder.prev_image()
        self.change_layer(self.curr_layer)
        if self.draw_rois:
            self.rois_canvas.draw_rois(self.im_folder.rois)

    @pyqtSlot(int)
    def bad_frame(self, state):
        if state == 0:
            # unchecked, mark frame as not bad
            pass
        else:
            # checked, mark frame as bad
            pass

    @pyqtSlot(int)
    def interesting_frame(self, state):
        pass


class MainMenu(QtWidgets.QMenuBar):
    sgnl_im_folder = pyqtSignal(ImageFolder)

    def __init__(self, parent):
        super(MainMenu, self).__init__(parent)

        self.image_folder = None
        file_menu = self.addMenu('&File')

        action_open = QtWidgets.QAction('&Open folder', self)
        action_open.triggered.connect(self.call_open)

        file_menu.addAction(action_open)

    def call_open(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.Directory)
        if dlg.exec_():
            folder = dlg.selectedFiles()[0]
            self.image_folder = ImageFolder(folder)
            self.sgnl_im_folder.emit(self.image_folder)


class BottomButtons(QtWidgets.QWidget):
    sgnl_change_frame = pyqtSignal(int)

    def __init__(self, parent=None):
        super(BottomButtons, self).__init__(parent)
        bb_layout = QtWidgets.QGridLayout(self)

        btn_play = QtWidgets.QPushButton('Play')
        btn_pause = QtWidgets.QPushButton('Pause')
        btn_prev = QtWidgets.QPushButton('Previous')
        btn_next = QtWidgets.QPushButton('Next')

        bb_layout.addWidget(btn_play, 0, 0)
        bb_layout.addWidget(btn_pause, 0, 1)
        bb_layout.addWidget(btn_prev, 1, 0)
        bb_layout.addWidget(btn_next, 1, 1)

        btn_prev.clicked.connect(lambda: self.call_change(-1))
        btn_next.clicked.connect(lambda: self.call_change(1))

    def call_change(self, direction):
        self.sgnl_change_frame.emit(direction)


class RightButtons(QtWidgets.QWidget):
    sgnl_change_im_layer = pyqtSignal(int)
    sgnl_toggle_rois = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(RightButtons, self).__init__(parent)
        rb_layout = QtWidgets.QGridLayout(self)

        # Column 0
        lbl_layers = QtWidgets.QLabel('Layers')
        lbl_layers.setAlignment(Qt.AlignCenter)
        self.btn_raw_im = QtWidgets.QPushButton('Raw Image')
        self.btn_bg_sub = QtWidgets.QPushButton('Background Subtracted')
        self.btn_bg_im = QtWidgets.QPushButton('Background')
        self.btn_bm_im = QtWidgets.QPushButton('Binary Mask')
        self.btn_rois = QtWidgets.QPushButton('RoIs')
        for btn in [self.btn_raw_im, self.btn_bg_sub, self.btn_bg_im, self.btn_bm_im, self.btn_rois]:
            btn.setCheckable(True)

        self.btn_raw_im.clicked.connect(lambda: self.call_btn(0))
        self.btn_bg_im.clicked.connect(lambda: self.call_btn(1))
        self.btn_bm_im.clicked.connect(lambda: self.call_btn(2))
        self.btn_bg_sub.clicked.connect(lambda: self.call_btn(3))
        self.btn_rois.toggled.connect(self.call_rois)

        rb_layout.addWidget(lbl_layers, 0, 0)
        rb_layout.addWidget(self.btn_raw_im, 1, 0)
        rb_layout.addWidget(self.btn_bg_im, 3, 0)
        rb_layout.addWidget(self.btn_bg_sub, 2, 0)
        rb_layout.addWidget(self.btn_bm_im, 4, 0)
        rb_layout.addWidget(self.btn_rois, 5, 0)

        # Column 1
        lbl_paint = QtWidgets.QLabel('Paint Tools')
        lbl_paint.setAlignment(Qt.AlignCenter)
        btn_paint = QtWidgets.QPushButton('Paintbrush')
        btn_fill = QtWidgets.QPushButton('Fill')
        btn_erase = QtWidgets.QPushButton('Erase')
        btn_erase.setCheckable(True)

        rb_layout.addWidget(lbl_paint, 0, 1)
        rb_layout.addWidget(btn_paint, 1, 1)
        rb_layout.addWidget(btn_fill, 2, 1)
        rb_layout.addWidget(btn_erase, 3, 1)

    def call_btn(self, idx):
        self.sgnl_change_im_layer.emit(idx)
        self.uncheck_others(idx)

    def call_rois(self, checked):
        self.sgnl_toggle_rois.emit(checked)

    def uncheck_others(self, btn):
        buttons = [self.btn_raw_im, self.btn_bg_im, self.btn_bm_im, self.btn_bg_sub]
        for i in range(len(buttons)):
            if i != btn:
                buttons[i].setChecked(False)


class BottomRightButtons(QtWidgets.QWidget):
    # We want check boxes for: bad frame, frame of interest
    # Maybe also for more than one fish in frame? Text box that takes a number?
    sgnl_cb_bad_changed = pyqtSignal(int)
    sgnl_cb_interest_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super(BottomRightButtons, self).__init__(parent)
        brb_layout = QtWidgets.QGridLayout(self)

        cb_bad = QtWidgets.QCheckBox('Bad frame')
        cb_interest = QtWidgets.QCheckBox('Interesting frame')

        brb_layout.addWidget(cb_bad, 0, 0)
        brb_layout.addWidget(cb_interest, 1, 0)

        cb_bad.stateChanged.connect(self.call_cb_bad)
        cb_interest.stateChanged.connect(self.call_cb_interest)

    def call_cb_bad(self, state):
        self.sgnl_cb_bad_changed.emit(state)

    def call_cb_interest(self, state):
        self.sgnl_cb_interest_changed.emit(state)


class MainCanvas(QtWidgets.QLabel):
    def __init__(self, parent):
        super(MainCanvas, self).__init__(parent)
        self._w, self._h = 1224, 425
        self.setMinimumSize(QSize(self._w, self._h))
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self._set_canvas()

    def _set_canvas(self):
        canvas = QPixmap(self._w, self._h)
        canvas.fill(QColor('transparent'))
        self.setPixmap(canvas)


class RoIsCanvas(MainCanvas):
    def __init__(self, parent):
        super(RoIsCanvas, self).__init__(parent)

        self.pen_color = QColor('#77FF0000')

    def draw_rois(self, rois):
        self.erase_rois()
        painter = QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(2)
        p.setColor(self.pen_color)
        painter.setPen(p)
        for roi in rois:
            roi_int = list(map(int, roi.split(',')))
            roi_qrect = QRect(roi_int[0] / 2, roi_int[1] / 2, (roi_int[2] - roi_int[0]) / 2, (roi_int[3] - roi_int[1]) / 2)
            painter.drawRect(roi_qrect)
        painter.end()
        self.update()

    def erase_rois(self):
        painter = QPainter(self.pixmap())
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        extents = self.pixmap().rect()
        painter.eraseRect(extents)
        painter.end()
        self.update()


class PaintingCanvas(MainCanvas):
    def __init__(self, parent):
        super(PaintingCanvas, self).__init__(parent)

        self.last_x, self.last_y = None, None
        self.pen_color = QColor('#FF0000')

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return  # Ignore the first time.

        painter = QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(8)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def erase_all(self):
        painter = QPainter(self.pixmap())
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        extents = self.pixmap().rect()
        painter.eraseRect(extents)
        painter.end()
        self.update()


class ImageFrame(MainCanvas):
    def __init__(self, parent):
        super(ImageFrame, self).__init__(parent)
        self.image = None  # im.scaled(1224, 425, Qt.KeepAspectRatioByExpanding)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        if self.image is not None:
            painter.drawPixmap(0, 0, self.image)
        painter.end()

    def update_image(self, im):
        self.image = im.scaled(self._w, self._h, Qt.KeepAspectRatioByExpanding)
        self.update()


def run():
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    run()