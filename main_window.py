from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QGridLayout, QLabel, QMainWindow, QWidget

from buttons import *
from canvases import *
from menus import *


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.im_folder = None
        self.curr_layer = 0
        self.draw_rois = False

        # Menu bar
        main_menu = MainMenu(self)
        self.setMenuWidget(main_menu)

        self.image_frame = ImageFrame(self)
        self.paint_canvas = PaintingCanvas(self)
        self.rois_canvas = RoIsCanvas(self)
        self.lbl_frame = QLabel('\n')
        self.lbl_frame_no = QLabel('\n')
        self.lbl_frame_no.setAlignment(Qt.AlignRight)

        # Set up layout of window inside central placeholder widget
        placeholder = QWidget()
        self.setCentralWidget(placeholder)
        grid_layout = QGridLayout(placeholder)
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
