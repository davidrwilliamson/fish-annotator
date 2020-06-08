from enum import IntEnum
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QAction, QFileDialog, QMenuBar, QWidget
from image_folder import ImageFolder


class ExportMenu(IntEnum):
    PREVIEW_ROIS = 0
    EXPORT_ROIS = 1
    EXPORT_MONTAGE = 2


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

        action_preview_rois = QAction('&Preview RoIs', self)
        action_preview_rois.triggered.connect(lambda: self._call_export(ExportMenu.PREVIEW_ROIS))
        action_export_rois = QAction('Export &RoIs', self)
        action_export_rois.triggered.connect(lambda: self._call_export(ExportMenu.EXPORT_MONTAGE))
        action_export_montage = QAction('Export &montage', self)
        action_export_montage.triggered.connect(lambda: self._call_export(ExportMenu.EXPORT_ROIS))

        export_menu.addAction(action_preview_rois)
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
    def _get_max_roi(im_folder: ImageFolder) -> list:
        curr_frame = im_folder.framepos[0]

        left, top, right, bottom = 9999, 9999, 0, 0

        i = 0
        im_folder.go_to_frame(i)

        while i < im_folder.num_frames:
            for roi in im_folder.rois:
                roi = list(map(int, roi.split(',')))
                if roi[0] < left:
                    left = roi[0]
                if roi[1] < top:
                    top = roi[1]
                if roi[2] > right:
                    right = roi[2]
                if roi[3] > bottom:
                    bottom = roi[3]
            im_folder.next_frame()
            i += 1

        im_folder.go_to_frame(curr_frame)
        max_roi = [left, bottom, right, top]

        return max_roi

    def preview_rois(self, im_folder: ImageFolder) -> None:
        all_rois = self._get_max_roi(im_folder)

    def export_rois(self, im_folder: ImageFolder) -> None:
        pass

    def export_montage(self, im_folder: ImageFolder) -> None:
        pass

