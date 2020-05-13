from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize, Qt, pyqtSignal, pyqtSlot
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

        self._curr_frame_no = 0
        self._no_of_frames = 0

        self._list_image_files()

    def _list_image_files(self):
        rois_filename = os.path.join(self.folder, 'analysis/RoIs')

        if os.path.isfile(rois_filename):
            rois_file = open(rois_filename, 'rt')
        else:
            print('{}: RoIs file missing.'.format(self.folder))
            return

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
    def curr_framepos(self):
        cf_no = self._curr_frame_no
        cf_fn = self._im_files[self._curr_frame_no]

        return cf_no, cf_fn

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

        main_menu = MainMenu(self)

        self.image_frame = ImageFrame(self)
        self.paint_canvas = PaintingCanvas(self)
        self.lbl_frame = QtWidgets.QLabel()
        self.lbl_frame_no = QtWidgets.QLabel()
        self.lbl_frame_no.setAlignment(Qt.AlignRight)

        placeholder = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(placeholder)
        grid_layout.addWidget(self.lbl_frame, 0, 0, 1, 1)
        grid_layout.addWidget(self.lbl_frame_no, 0, 0, 1, 1)
        grid_layout.addWidget(self.image_frame, 1, 0)
        grid_layout.addWidget(self.paint_canvas, 1, 0)

        bottom_buttons = BottomButtons()
        grid_layout.addWidget(bottom_buttons, 2, 0)

        right_buttons = RightButtons()
        grid_layout.addWidget(right_buttons, 1, 1)

        # self.setCentralWidget(self.label)
        self.setCentralWidget(placeholder)
        self.setMenuWidget(main_menu)
        self.setWindowTitle("Fish Annotator")

        main_menu.sgnl_im_folder.connect(self.set_im_folder)
        right_buttons.sgnl_change_im_layer.connect(self.change_layer)
        bottom_buttons.sgnl_change_frame.connect(self.change_frame)

    def update_lbl_frame(self):
        folder = self.im_folder.folder
        cf_no, cf_fn = self.im_folder.curr_framepos
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

    @pyqtSlot(int)
    def change_frame(self, direction):
        if direction > 0:
            self.im_folder.next_image()
        if direction < 0:
            self.im_folder.prev_image()
        self.change_layer(self.curr_layer)


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

        btn_prev.clicked.connect(self.call_prev)
        btn_next.clicked.connect(self.call_next)

    def call_prev(self):
        self.sgnl_change_frame.emit(-1)

    def call_next(self):
        self.sgnl_change_frame.emit(1)


class RightButtons(QtWidgets.QWidget):
    sgnl_change_im_layer = pyqtSignal(int)
    sgnl_toggle_rois = pyqtSignal()

    def __init__(self, parent=None):
        super(RightButtons, self).__init__(parent)
        rb_layout = QtWidgets.QGridLayout(self)

        # Column 0
        lbl_layers = QtWidgets.QLabel('Layers')
        lbl_layers.setAlignment(Qt.AlignCenter)
        btn_raw_im = QtWidgets.QPushButton('Raw Image')
        btn_bg_sub = QtWidgets.QPushButton('Background Subtracted')
        btn_bg_im = QtWidgets.QPushButton('Background')
        btn_bm_im = QtWidgets.QPushButton('Binary Mask')
        btn_rois = QtWidgets.QPushButton('RoIs')
        btn_rois.setCheckable(True)

        btn_raw_im.clicked.connect(self.call_raw_im)
        btn_bg_im.clicked.connect(self.call_bg_im)
        btn_bm_im.clicked.connect(self.call_bm_im)
        btn_bg_sub.clicked.connect(self.call_bg_sub)

        rb_layout.addWidget(lbl_layers, 0, 0)
        rb_layout.addWidget(btn_raw_im, 1, 0)
        rb_layout.addWidget(btn_bg_im, 3, 0)
        rb_layout.addWidget(btn_bg_sub, 2, 0)
        rb_layout.addWidget(btn_bm_im, 4, 0)
        rb_layout.addWidget(btn_rois, 5, 0)

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

    def call_raw_im(self):
        self.sgnl_change_im_layer.emit(0)

    def call_bg_im(self):
        self.sgnl_change_im_layer.emit(1)

    def call_bm_im(self):
        self.sgnl_change_im_layer.emit(2)

    def call_bg_sub(self):
        self.sgnl_change_im_layer.emit(3)


class BottomRightButtons(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(BottomRightButtons, self).__init__(parent)
        brb_layout = QtWidgets.QGridLayout(self)


class PaintingCanvas(QtWidgets.QLabel):
    def __init__(self, parent):
        super(PaintingCanvas, self).__init__(parent)
        self.setMinimumSize(QSize(1224, 425))
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

        canvas = QPixmap(1224, 425)
        canvas.fill(QColor('transparent'))
        self.setPixmap(canvas)

        self.last_x, self.last_y = None, None
        self.pen_color = QColor('#FF0000')

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

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

    # def draw_something(self):
    #     painter = QPainter(self.label.pixmap())
    #     pen = QPen()
    #     pen.setWidth(3)
    #     pen.setColor(QColor('#EB5160'))
    #     painter.setPen(pen)
    #     painter.drawRect(50, 50, 100, 100)
    #     painter.drawRect(60, 60, 150, 100)
    #     painter.end()


class ImageFrame(QtWidgets.QWidget):
    def __init__(self, parent):
        super(ImageFrame, self).__init__(parent)
        self.setMinimumSize(QSize(1224, 425))
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.image = None  # im.scaled(1224, 425, Qt.KeepAspectRatioByExpanding)

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image is not None:
            qp.drawPixmap(0, 0, self.image)
        qp.end()

    def update_image(self, im):
        self.image = im.scaled(1224, 425, Qt.KeepAspectRatioByExpanding)
        self.update()


def run():
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    run()