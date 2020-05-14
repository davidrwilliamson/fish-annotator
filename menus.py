from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QAction, QFileDialog, QMenuBar, QWidget
from image_folder import ImageFolder


class MainMenu(QMenuBar):
    sgnl_im_folder = pyqtSignal(ImageFolder)

    def __init__(self, parent: QWidget) -> None:
        super(MainMenu, self).__init__(parent)

        self.image_folder = None
        file_menu = self.addMenu('&File')

        action_open = QAction('&Open folder', self)
        action_open.triggered.connect(self.call_open)

        file_menu.addAction(action_open)

    def call_open(self) -> None:
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            folder = dlg.selectedFiles()[0]
            self.image_folder = ImageFolder(folder)
            self.sgnl_im_folder.emit(self.image_folder)
