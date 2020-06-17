from enum import IntEnum
import os
from PyQt5.QtCore import pyqtSignal, QRect
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import QAction, QFileDialog, QLabel, QMenuBar, QWidget
from typing import Tuple

from image_folder import ImageFolder


class ExportMenu(IntEnum):
    PREVIEW_ROIS = 0
    EXPORT_ROIS = 1
    EXPORT_MONTAGE = 2


# class PreviewPopup(QWidget):
#     def __init__(self) -> None:
#         QWidget.__init__(self)
#         self.pixmap = None
#         self.roi: QRect = None
#
#     def paintEvent(self, e) -> None:
#         painter = QPainter(self)
#         painter.begin(self)
#         if self.pixmap:
#             dest_rect = QRect(self.rect().center().x() - (self.roi.width() / 2),
#                               self.rect().center().y() - (self.roi.width() / 2), self.roi.width(), self.roi.height())
#             painter.drawPixmap(dest_rect, self.pixmap, self.roi)
#         painter.end()
#
#     def draw_pixmap(self, pixmap: QPixmap, roi: QRect):
#         self.pixmap = pixmap
#         self.roi = roi
#         self.update()


class MainMenu(QMenuBar):
    sgnl_im_folder = pyqtSignal(ImageFolder)
    sgnl_export_menu = pyqtSignal(int)

    def __init__(self, parent: QWidget) -> None:
        super(MainMenu, self).__init__(parent)

        self.image_folder = None
        file_menu = self.addMenu('&File')

        action_open = QAction('&Open folder', self)
        action_open.triggered.connect(self._call_open)

        file_menu.addAction(action_open)

        export_menu = self.addMenu('&Export')

        # action_preview_rois = QAction('&Preview RoIs', self)
        # action_preview_rois.triggered.connect(lambda: self._call_export(ExportMenu.PREVIEW_ROIS))
        action_export_rois = QAction('Export &RoIs', self)
        action_export_rois.triggered.connect(lambda: self._call_export(ExportMenu.EXPORT_ROIS))
        action_export_montage = QAction('Export &montage', self)
        action_export_montage.triggered.connect(lambda: self._call_export(ExportMenu.EXPORT_MONTAGE))

        # export_menu.addAction(action_preview_rois)
        export_menu.addAction(action_export_rois)
        export_menu.addAction(action_export_montage)

    def _call_open(self) -> None:
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            folder = dlg.selectedFiles()[0]
            self.image_folder = ImageFolder(folder)
            self.sgnl_im_folder.emit(self.image_folder)

    def _call_export(self, option: IntEnum) -> None:
        self.sgnl_export_menu.emit(option)

    @staticmethod
    def _get_max_roi(im_folder: ImageFolder) -> Tuple[int, int]:
        width, height = 0, 0

        for frame in im_folder.frames:
            for roi in im_folder.rois:
                roi = list(map(int, roi.split(',')))
                width = roi[2] - roi[0] if roi[2] - roi[0] > width else width
                height = roi[3] - roi[1] if roi[3] - roi[1] > height else height

        return width, height

    @staticmethod
    def _get_im_raw(im_folder: ImageFolder) -> Tuple[QPixmap, QRect]:
        im_raw = im_folder.curr_frames[0]
        rois = im_folder.rois

        roi = rois[0]
        roi = list(map(int, roi.split(',')))
        roi = QRect(roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1])
        return im_raw, roi

    # def preview_rois(self, im_folder: ImageFolder) -> PreviewPopup:
    #     popup = PreviewPopup()
    #     w, h = self._get_max_roi(im_folder)
    #     popup.setGeometry(QRect(100, 100, w, h))
    #     popup.show()
    #
    #     return popup
    #
    # def update_preview(self, im_folder: ImageFolder, popup: PreviewPopup) -> None:
    #     f = im_folder.frames
    #
    #     im_raw, roi = self._get_im_raw(im_folder)
    #     popup.draw_pixmap(im_raw, roi)
    #     next(f)

    def export_rois(self, im_folder: ImageFolder) -> None:
        try:
            os.mkdir(os.path.join(im_folder.folder, 'png_exports'))
        except FileExistsError:
            print('Directory {} already exists.'.format(os.path.join(im_folder.folder, 'png_exports')))

        for frame in im_folder.frames:
            im_raw, roi = self._get_im_raw(im_folder)
            im_save = im_raw.copy(roi)

            save_path = os.path.join(im_folder.folder, 'png_exports', '{}.png'.format(im_folder.framepos[1]))
            im_save.save(save_path, 'png')

    def export_montage(self, im_folder: ImageFolder) -> None:
        pass


