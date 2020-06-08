from PyQt5.QtCore import pyqtSlot, QBuffer, QIODevice
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QGridLayout, QLabel, QMainWindow, QWidget
from typing import List

from buttons import *
from canvases import *
from menus import *


class MainWindow(QMainWindow):
    def __init__(self, parent: QMainWindow = None) -> None:
        super(MainWindow, self).__init__(parent)

        self.im_folder: ImageFolder = None
        self.curr_layer: int = 0
        self.curr_ann_layer: int = -1
        self.draw_rois: bool = False

        # Menu bar
        self.main_menu = MainMenu(self)
        self.setMenuWidget(self.main_menu)

        self.image_frame = ImageFrame(self)
        self.rois_canvas = RoIsCanvas(self)
        self.lbl_frame = QLabel('\n')
        self.lbl_frame_no = QLabel('\n')
        self.lbl_frame_no.setAlignment(Qt.AlignRight)

        self.scale_bar = ScaleBar(self.image_frame, 1)
        self.scale_bar.setVisible(False)

        self.saved_canvases: List[QBuffer] = []
        self.annotation_canvases: List[PaintingCanvas] = []
        self.annotation_canvases.append(PaintingCanvas(self, 'red'))
        self.annotation_canvases.append(PaintingCanvas(self, 'blue'))
        self.annotation_canvases.append(PaintingCanvas(self, 'green'))

        self.change_ann_layer(False, self.curr_ann_layer)  # Disables and hides annotation canvases

        # Set up layout of window inside central placeholder widget
        placeholder = QWidget()
        self.setCentralWidget(placeholder)
        grid_layout = QGridLayout(placeholder)
        grid_layout.addWidget(self.lbl_frame, 0, 0)
        grid_layout.addWidget(self.lbl_frame_no, 0, 0)
        grid_layout.addWidget(self.image_frame, 1, 0)
        grid_layout.addWidget(self.rois_canvas, 1, 0)
        for canvas in self.annotation_canvases:
            grid_layout.addWidget(canvas, 1, 0)

        # Add various sets of buttons
        self.bottom_buttons = BottomButtons()
        grid_layout.addWidget(self.bottom_buttons, 2, 0)
        self.right_buttons = RightButtons()
        grid_layout.addWidget(self.right_buttons, 1, 1)
        self.br_buttons = BottomRightButtons()
        grid_layout.addWidget(self.br_buttons, 2, 1)

        # Connect up button signals
        self.main_menu.sgnl_im_folder.connect(self.set_im_folder)
        self.main_menu.sgnl_export_menu.connect(self.export_menu)
        self.right_buttons.sgnl_change_im_layer.connect(self.change_im_layer)
        self.right_buttons.sgnl_change_ann_layer.connect(self.change_ann_layer)
        self.right_buttons.sgnl_toggle_rois.connect(self.toggle_rois)
        self.right_buttons.sgnl_adjust_brush_size.connect(self.adjust_brush_size)
        self.bottom_buttons.sgnl_change_frame.connect(self.change_frame)
        self.bottom_buttons.sgnl_adjust_brightness.connect(self.adjust_brightness)
        self.bottom_buttons.sgnl_adjust_contrast.connect(self.adjust_contrast)
        self.bottom_buttons.sgnl_toggle_scale_bar.connect(self.toggle_scale_bar)
        self.br_buttons.sgnl_bad_frame.connect(self.bad_frame)
        self.br_buttons.sgnl_interesting_frame.connect(self.interesting_frame)

        self.setWindowTitle("Fish Annotator")

    def update_lbl_frame(self) -> None:
        folder = self.im_folder.folder
        cf_no, cf_fn = self.im_folder.framepos
        num_frames = self.im_folder.num_frames
        self.lbl_frame.setText('Folder:  {}\nFile:      {}'.format(folder, cf_fn))
        self.lbl_frame_no.setText('\nFrame: {} / {}'.format(cf_no, num_frames))

    def save_annotations(self) -> None:
        canvases = [None] * len(self.annotation_canvases)
        save: bool = False
        i = 0
        for canvas in self.annotation_canvases:
            if canvas.is_used:  # Don't bother to save unless something has actually been drawn
                buffer = QBuffer()
                buffer.open(QIODevice.ReadWrite)
                canvas.pixmap().save(buffer, 'png')
                buffer.close()
                canvases[i] = buffer
                canvas.erase_all()
                save = True
            i += 1
        if save:
            self.saved_canvases[self.im_folder.framepos[0]] = canvases

    def load_annotations(self) -> None:
        k = self.im_folder.framepos[0]
        # If the current frame has ANY annotations on it, on any layer...
        if self.saved_canvases[k]:
            for i in range(len(self.saved_canvases[k])):
                # If this specific layer has an annotation on it, display that annotation
                if self.saved_canvases[k][i]:
                    buffer = self.saved_canvases[k][i].buffer()
                    self.annotation_canvases[i].pixmap().loadFromData(buffer)
                # Otherwise clear the layer so we aren't showing annotations from other frames
                else:
                    self.annotation_canvases[i].erase_all()
                self.annotation_canvases[i].update()
        # Clear all layers if this frame doesn't have any annotations
        else:
            for canvas in self.annotation_canvases:
                canvas.erase_all()

    @pyqtSlot(ImageFolder)
    def set_im_folder(self, im_folder: ImageFolder) -> None:
        self.im_folder = im_folder
        self.saved_canvases = [None for i in range(im_folder.num_frames)]
        self.change_im_layer(0)
        self.right_buttons.enable_buttons(selection=range(8))
        self.right_buttons.uncheck_others(self.right_buttons.btns_im_layers, 0)
        self.bottom_buttons.enable_buttons()
        self.br_buttons.enable_buttons()

    @pyqtSlot(int)
    def change_im_layer(self, im_idx: int) -> None:
        curr_frames = self.im_folder.curr_frames
        im = curr_frames[im_idx]
        self.curr_layer = im_idx
        self.image_frame.update_image(im)
        self.update_lbl_frame()

    @pyqtSlot(bool, int)
    def change_ann_layer(self, checked: bool, ann_idx: int) -> None:
        for canvas in self.annotation_canvases:
            canvas.setEnabled(False)
            canvas.setVisible(False)
            # self.right_buttons.enable_buttons(False, range(9, 11))
        if ann_idx >= 0:  # This means we won't try to set the brush image before a canvas exists
            self.right_buttons.uncheck_others(self.right_buttons.btns_painting, -1)
            self.right_buttons.enable_buttons(False, selection=range(8, 13))
            self.curr_ann_layer = ann_idx
            self.right_buttons.set_lbl_curr_brush(self.annotation_canvases[ann_idx], checked)
        if checked:  # Lets us hide annotations again by unchecking button
            self.right_buttons.btn_paint.setChecked(True)
            self.annotation_canvases[ann_idx].setEnabled(True)
            self.annotation_canvases[ann_idx].setVisible(True)
            self.right_buttons.enable_buttons(selection=range(8, 13))

    @pyqtSlot(bool)
    def toggle_rois(self, checked: bool) -> None:
        self.draw_rois = checked
        if checked:
            rois = self.im_folder.rois
            self.rois_canvas.draw_rois(rois)
        else:
            self.rois_canvas.erase_rois()

    @pyqtSlot(int)
    def change_frame(self, direction: int) -> None:
        self.save_annotations()
        if direction > 0:
            self.im_folder.next_frame()
        if direction < 0:
            self.im_folder.prev_frame()
        self.change_im_layer(self.curr_layer)
        if self.draw_rois:
            self.rois_canvas.draw_rois(self.im_folder.rois)
        self.load_annotations()

    @pyqtSlot(int)
    def adjust_brightness(self, value: int) -> None:
        self.image_frame.set_brightness(value)

    @pyqtSlot(int)
    def adjust_contrast(self, value: int) -> None:
        self.image_frame.set_contrast(value)

    @pyqtSlot(int)
    def adjust_brush_size(self, value: int) -> None:
        for canvas in self.annotation_canvases:
            canvas.pen_size = value
        self.right_buttons.set_lbl_curr_brush(self.annotation_canvases[self.curr_ann_layer], True)

    @pyqtSlot(int)
    def bad_frame(self, state: int) -> None:
        if state == 0:
            # unchecked, mark frame as not bad
            pass
        else:
            # checked, mark frame as bad
            pass

    @pyqtSlot(int)
    def interesting_frame(self, state: int) -> None:
        pass

    @pyqtSlot(int)
    def toggle_scale_bar(self, state: int) -> None:
        if state == 0:
            self.scale_bar.setVisible(False)
        else:
            self.scale_bar.setVisible(True)

    @pyqtSlot(int)
    def export_menu(self, option: IntEnum) -> None:
        if option == ExportMenu.PREVIEW_ROIS:
            self.main_menu.preview_rois(self.im_folder)
        elif option == ExportMenu.EXPORT_ROIS:
            self.main_menu.export_rois(self.im_folder)
        elif option == ExportMenu.EXPORT_MONTAGE:
            self.main_menu.export_montage(self.im_folder)


class ScaleBar(QLabel):
    def __init__(self, parent: QWidget, scale: float) -> None:
        super(ScaleBar, self).__init__(parent)

        # Default scale of 1px ~ 1um, adjustable here
        self.scale_factor = scale

        self.w, self.h = 190, 50
        self.setGeometry(parent.width() - self.w, parent.height() - self.h, self.w, self.h)
        self.setMinimumSize(QSize(self.w, self.h))
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self._set_canvas()
        self._draw_bars()
        self._draw_lables()

    def _set_canvas(self) -> None:
        canvas = QPixmap(self.w, self.h)
        canvas.fill(QColor('transparent'))
        # canvas.fill(QColor('red'))
        self.setPixmap(canvas)

    def _draw_bars(self) -> None:
        painter = QPainter(self.pixmap())
        painter.setPen(QPen(QColor('black'),  1, Qt.SolidLine))
        painter.setBrush(QColor('white'))
        # Mot filled
        painter.drawRect(10, 25, 10, 10)
        painter.drawRect(20, 15, 10, 10)
        painter.drawRect(30, 25, 10, 10)
        painter.drawRect(40, 15, 10, 10)
        painter.drawRect(50, 25, 40, 10)
        painter.drawRect(90, 15, 40, 10)
        painter.drawRect(130, 25, 40, 10)

        painter.setBrush(QColor('black'))
        # Filled
        painter.drawRect(10, 15, 10, 10)
        painter.drawRect(20, 25, 10, 10)
        painter.drawRect(30, 15, 10, 10)
        painter.drawRect(40, 25, 10, 10)
        painter.drawRect(50, 15, 40, 10)
        painter.drawRect(90, 25, 40, 10)
        painter.drawRect(130, 15, 40, 10)

    def _draw_lables(self) -> None:
        lab_0 = QLabel(self)
        lab_0.setText('0')
        lab_0.setAlignment(Qt.AlignCenter)
        lab_0.setGeometry(-5, 0, 30, 10)

        lab_25 = QLabel(self)
        lab_25.setText('{}'.format(40 * self.scale_factor))
        lab_25.setAlignment(Qt.AlignCenter)
        lab_25.setGeometry(35, 0, 30, 10)

        lab_50 = QLabel(self)
        lab_50.setText('{}'.format(80 * self.scale_factor))
        lab_50.setAlignment(Qt.AlignCenter)
        lab_50.setGeometry(75, 0, 30, 10)

        lab_100 = QLabel(self)
        lab_100.setText('{}'.format(160 * self.scale_factor))
        lab_100.setAlignment(Qt.AlignCenter)
        lab_100.setGeometry(155, 0, 30, 10)

        lab_um = QLabel(self)
        lab_um.setText('micrometres')
        lab_um.setAlignment(Qt.AlignCenter)
        lab_um.setGeometry(45, 40, 90, 10)