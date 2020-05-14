from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtGui import QPainter, QPixmap, QColor
from PyQt5.QtWidgets import QLabel, QSizePolicy


class MainCanvas(QLabel):
    def __init__(self, parent):
        super(MainCanvas, self).__init__(parent)
        self._w, self._h = 1224, 425
        self.setMinimumSize(QSize(self._w, self._h))
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
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
