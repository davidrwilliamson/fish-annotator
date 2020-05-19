from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtGui import QColor, QImage, QMouseEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy, QWidget
import qimage2ndarray as q2n
import numpy as np
from cv2 import convertScaleAbs


class MainCanvas(QLabel):
    def __init__(self, parent: QWidget) -> None:
        super(MainCanvas, self).__init__(parent)
        self._w, self._h = 1224, 425
        self.setMinimumSize(QSize(self._w, self._h))
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self._set_canvas()

    def _set_canvas(self) -> None:
        canvas = QPixmap(self._w, self._h)
        canvas.fill(QColor('transparent'))
        self.setPixmap(canvas)


class RoIsCanvas(MainCanvas):
    def __init__(self, parent: QWidget) -> None:
        super(RoIsCanvas, self).__init__(parent)

        self.pen_colour = QColor('#77FF0000')

    def draw_rois(self, rois: list) -> None:
        self.erase_rois()
        painter = QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(2)
        p.setColor(self.pen_colour)
        painter.setPen(p)
        for roi in rois:
            roi_int = list(map(int, roi.split(',')))
            roi_qrect = QRect(roi_int[0] / 2, roi_int[1] / 2,
                              (roi_int[2] - roi_int[0]) / 2, (roi_int[3] - roi_int[1]) / 2)
            painter.drawRect(roi_qrect)
        painter.end()
        self.update()

    def erase_rois(self) -> None:
        painter = QPainter(self.pixmap())
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        extents = self.pixmap().rect()
        painter.eraseRect(extents)
        painter.end()
        self.update()


class PaintingCanvas(MainCanvas):
    def __init__(self, parent: QWidget, colour: str) -> None:
        super(PaintingCanvas, self).__init__(parent)

        self.last_x, self.last_y = None, None
        self.pen_colour = QColor(colour)
        self.pen_size = 5
        self.is_used = False

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self.last_x is None:  # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return  # Ignore the first time.

        painter = QPainter(self.pixmap())
        # Would like to have semi-transparent annotation overlays without painting looking bad
        # p.setOpacity(0.5) after pen is set looks bad.
        # painter.setCompositionMode(QPainter.CompositionMode_Overlay) doesn't seem to do anything useful?
        p = painter.pen()
        p.setWidth(self.pen_size)
        p.setColor(self.pen_colour)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        self.is_used = True  # Mark if the canvas has anything drawn on it

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self.last_x = None
        self.last_y = None

    def erase_all(self) -> None:
        painter = QPainter(self.pixmap())
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        extents = self.pixmap().rect()
        painter.eraseRect(extents)
        painter.end()
        self.update()
        self.is_used = False  # Now that the canvas is empty we mark it as unused


class ImageFrame(MainCanvas):
    def __init__(self, parent: QWidget) -> None:
        super(ImageFrame, self).__init__(parent)
        self.image: QPixmap = None  # im.scaled(1224, 425, Qt.KeepAspectRatioByExpanding)

        self._adjusted_im: QPixmap = None
        self._brightness_adjustment = 0
        self._contrast_adjustment = 1

    def paintEvent(self, event) -> None:
        painter = QPainter()
        painter.begin(self)
        if self._adjusted_im is not None:
            painter.drawPixmap(0, 0, self._adjusted_im)
        elif self.image is not None:
            painter.drawPixmap(0, 0, self.image)
        painter.end()

    def _adjust_image(self) -> None:
        im: QImage = self.image.toImage()
        np_im = q2n.rgb_view(im)

        alpha: float = self._contrast_adjustment
        beta: int = self._brightness_adjustment
        np_im = convertScaleAbs(np_im, alpha=alpha, beta=beta)

        q_im = q2n.array2qimage(np_im)
        pmap = QPixmap.fromImage(q_im)
        self._adjusted_im = pmap
        self.update()

    def set_brightness(self, value: int) -> None:
        self._brightness_adjustment = value
        self._adjust_image()

    def set_contrast(self, value: int) -> None:
        self._contrast_adjustment = value / 10.0
        self._adjust_image()

    def update_image(self, im: QPixmap) -> None:
        self.image = im.scaled(self._w, self._h, Qt.KeepAspectRatioByExpanding)
        self._adjust_image()
        self.update()