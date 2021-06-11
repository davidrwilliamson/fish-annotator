from enum import IntEnum
import os
from PyQt5.QtCore import pyqtSignal, QPoint, QRect, Qt
from PyQt5.QtGui import QImage, QColor, QPixmap, QPainter
from PyQt5.QtWidgets import QAction, QFileDialog, QMenuBar, QWidget
from typing import Tuple
import qimage2ndarray as q2n
import numpy as np

from image_folder import ImageFolder
import errors


class ExportMenu(IntEnum):
    PREVIEW_ROIS = 0
    EXPORT_ROIS = 1
    EXPORT_MONTAGE = 2
    EXPORT_FULL = 3
    EXPORT_INTERESTING = 4
    EXPORT_CURRENT = 5


class AnalysisMenu(IntEnum):
    CIRCLES = 0
    BACKGROUNDER = 1


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
    sgnl_export_menu = pyqtSignal(IntEnum)
    sgnl_analysis_menu = pyqtSignal(IntEnum)
    sgnl_save_ann = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super(MainMenu, self).__init__(parent)

        self.image_folder = None
        file_menu = self.addMenu('&File')

        action_open = QAction('&Open folder', self)
        action_save = QAction('&Save annotations', self)
        action_open.triggered.connect(self._call_open)
        action_save.triggered.connect(self._call_save)

        file_menu.addAction(action_open)
        # file_menu.addAction(action_save)

        self.export_menu = self.addMenu('&Export')
        self.export_menu.setEnabled(False)
        self.export_menu.addAction(action_save)
        # action_preview_rois = QAction('&Preview RoIs', self)
        # action_preview_rois.triggered.connect(lambda: self._call_export(ExportMenu.PREVIEW_ROIS))
        action_export_rois = QAction('Export cropped &RoIs', self)
        action_export_rois.triggered.connect(lambda: self._call_export(ExportMenu.EXPORT_ROIS))
        action_export_full = QAction('Export all &full frames', self)
        action_export_full.triggered.connect(lambda: self._call_export(ExportMenu.EXPORT_FULL))
        action_export_interesting = QAction('Export only &interesting frames', self)
        action_export_interesting.triggered.connect(lambda: self._call_export(ExportMenu.EXPORT_INTERESTING))
        action_export_montage = QAction('Export &montage', self)
        action_export_montage.triggered.connect(lambda: self._call_export(ExportMenu.EXPORT_MONTAGE))
        action_export_current = QAction('Export &current image', self)
        action_export_current.triggered.connect(lambda: self._call_export(ExportMenu.EXPORT_CURRENT))

        # export_menu.addAction(action_preview_rois)
        self.export_menu.addAction(action_export_full)
        self.export_menu.addAction(action_export_interesting)
        self.export_menu.addAction(action_export_rois)
        self.export_menu.addAction(action_export_montage)
        self.export_menu.addAction(action_export_current)

        self.analysis_menu = self.addMenu('&Analysis')
        self.analysis_menu.setEnabled(False)

        action_analyse_circles = QAction('&Find egg circles (this frame)', self)
        action_analyse_circles.triggered.connect(lambda: self._call_analyse(AnalysisMenu.CIRCLES))
        action_analyse_background_subtract = QAction('Perform &Background subtraction and RoI extraction (all frames)',
                                                     self)
        action_analyse_background_subtract.triggered.connect(lambda: self._call_analyse(AnalysisMenu.BACKGROUNDER))

        self.analysis_menu.addAction(action_analyse_circles)
        self.analysis_menu.addAction(action_analyse_background_subtract)

    def _call_open(self) -> None:
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        # folder = dlg.getExistingDirectory(self, 'Choose folder :)', '/media', QFileDialog.ShowDirsOnly)
        if dlg.exec_():
            folder = dlg.selectedFiles()[0]
            try:
                self.image_folder = ImageFolder(folder)
            except errors.NoImageFilesError as e:
                print(e)
            else:
                self.sgnl_im_folder.emit(self.image_folder)

    def _call_save(self) -> None:
        self.sgnl_save_ann.emit()

    def _call_export(self, option: IntEnum) -> None:
        self.sgnl_export_menu.emit(option)

    def _call_analyse(self, option: IntEnum) -> None:
        self.sgnl_analysis_menu.emit(option)

    def enable_export(self) -> None:
        self.export_menu.setEnabled(True)
        self.analysis_menu.setEnabled(True)

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

    @staticmethod
    def _make_export_folder(folder, export_type: IntEnum) -> str:
        root_path = os.path.join(folder, 'png_exports')

        if export_type == ExportMenu.EXPORT_FULL:
            subfolder = os.path.join(root_path, 'full')
        elif export_type == ExportMenu.EXPORT_ROIS:
            subfolder = os.path.join(root_path, 'cropped')
        elif export_type == ExportMenu.EXPORT_MONTAGE:
            subfolder = os.path.join(root_path, 'montage')
        elif export_type == ExportMenu.EXPORT_INTERESTING:
            subfolder = os.path.join(root_path, 'interesting')
        elif export_type == ExportMenu.EXPORT_CURRENT:
            subfolder = os.path.join(root_path, 'selected')
        else:
            raise

        try:
            os.mkdir(root_path)
        except FileExistsError:
            pass
            # print('Directory {} already exists.'.format(os.path.join(root_path)))

        try:
            os.mkdir(subfolder)
        except FileExistsError:
            pass
            # print('Directory {} already exists.'.format(subfolder))

        return subfolder

    def export_rois(self, im_folder: ImageFolder) -> None:
        # TODO: Handle frames that don't have an RoI (this is a more general problem that
        #  ultimately needs to be handled better than just ignoring them here)
        save_folder = self._make_export_folder(im_folder.folder, ExportMenu.EXPORT_ROIS)

        for frame in im_folder.frames:
            im_raw, roi = self._get_im_raw(im_folder)
            im_save = im_raw.copy(roi)

            save_path = os.path.join(save_folder, '{}_cropped.png'.format(im_folder.framepos[1]))
            im_save.save(save_path, 'png')
        print('Saved images to: {}'.format(save_folder))

    def export_full_frames(self, im_folder: ImageFolder) -> None:
        save_folder = self._make_export_folder(im_folder.folder, ExportMenu.EXPORT_FULL)

        im_folder.toggle_show_interesting(True)
        im_folder.toggle_show_other(True)
        for frame in im_folder.frames:
            im_raw, _ = self._get_im_raw(im_folder)

            save_path = os.path.join(save_folder, '{}_full.png'.format(im_folder.framepos[1]))
            im_raw.save(save_path, 'png')
        print('Saved images to: {}'.format(save_folder))

    def export_montage(self, im_folder: ImageFolder) -> None:
        pass
        # save_folder = self._make_export_folder(im_folder.folder, ExportMenu.EXPORT_MONTAGE)

    def export_interesting(self, im_folder: ImageFolder) -> None:
        save_folder = self._make_export_folder(im_folder.folder, ExportMenu.EXPORT_INTERESTING)

        im_folder.toggle_show_interesting(True)
        im_folder.toggle_show_other(False)
        for frame in im_folder.frames:
            im_raw, _ = self._get_im_raw(im_folder)

            save_path = os.path.join(save_folder, '{}_full.png'.format(im_folder.framepos[1]))
            im_raw.save(save_path, 'png')
        print('Saved images to: {}'.format(save_folder))

    def export_current(self, im_folder: ImageFolder, image_frame: 'ImageFrame',
                       rois_canvas, painting_canvas, nn_preview_canvas) -> None:
        save_folder = self._make_export_folder(im_folder.folder, ExportMenu.EXPORT_CURRENT)
        n = 0
        save_path = os.path.join(save_folder, '{}_selected_view_{}.png'.format(im_folder.framepos[1], n))
        while os.path.isfile(save_path):
            save_path = os.path.join(save_folder, '{}_selected_view_{}.png'.format(im_folder.framepos[1], n))
            n += 1

        if image_frame._adjusted_im is not None:
            base_im: QImage = image_frame._adjusted_im.toImage()
        else:
            base_im: QImage = image_frame.image.toImage()

        im_size = base_im.size()
        im: QImage = base_im.copy()
        im = im.convertToFormat(QImage.Format_ARGB32)

        painter = QPainter(im)
        painter.setCompositionMode(painter.CompositionMode_SourceOver)
        for canvas in [rois_canvas, painting_canvas, nn_preview_canvas]:
            if canvas is not None:
                canvas_image: QImage = canvas.pixmap().toImage()
                painter.drawImage(QRect(QPoint(0, 0), im_size), canvas_image)
        painter.end()
        im.save(save_path, 'png')
        print('Saved image: {}'.format(save_path))