from PyQt5.QtCore import QPoint, QPointF, QRect, QSize, Qt
from PyQt5.QtGui import QColor, QImage, QMouseEvent, QPainter, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy, QWidget
import qimage2ndarray as q2n
import numpy as np
import cv2 as cv


class MainCanvas(QLabel):
    def __init__(self, parent: QWidget) -> None:
        super(MainCanvas, self).__init__(parent)
        self._w, self._h = 0, 0
        # self.set_frame_size(self._w, self._h)

    def _set_canvas(self) -> None:
        canvas = QPixmap(self._w, self._h)
        canvas.fill(QColor('transparent'))
        self.setPixmap(canvas)

    def set_frame_size(self, w, h) -> None:
        self._w, self._h = w / 2, h / 2
        self.setMinimumSize(QSize(self._w, self._h))
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self._set_canvas()


class RoIsCanvas(MainCanvas):
    def __init__(self, parent: QWidget) -> None:
        super(RoIsCanvas, self).__init__(parent)

        self.pen_colour = QColor('#77FF0000')

    def draw_rois(self, rois: list) -> None:
        self.erase_rois()

        if rois:
            painter = QPainter(self.pixmap())
            p = painter.pen()
            p.setWidth(2)
            p.setColor(self.pen_colour)
            painter.setPen(p)
            for roi in rois:
                roi_int = list(map(int, roi.split(',')))
                # RoI given as distances from top-left corner (0, 0): [left, top, right, bottom]
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

    def paintEvent(self, event) -> None:
        if self._w > 0 and self._h > 0:
            painter = QPainter()
            painter.begin(self)

            painter.drawPixmap(0, 0, self.pixmap())
            painter.end()


class PaintingCanvas(MainCanvas):
    def __init__(self, parent: QWidget, colour: str) -> None:
        super(PaintingCanvas, self).__init__(parent)

        self.last_x, self.last_y = None, None
        self.pen_colour = QColor(colour)
        self.pen_size = 3
        self.is_used = False
        self.is_cleared = False
        self.brush_erase = False

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
        if self.brush_erase:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            p.setColor(QColor('transparent'))
        else:
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
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

    def draw_circle(self, circle) -> None:
        x, y, r = (circle[0] / 2), (circle[1] / 2), (circle[2] / 2)
        c = QPointF(x, y)
        painter = QPainter(self.pixmap())
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p = painter.pen()
        p.setColor(self.pen_colour)
        p.setWidth(1)
        painter.setPen(p)
        painter.drawEllipse(c, r, r)
        painter.end()
        self.update()

        self.is_used = True

    def paintEvent(self, event) -> None:
        if self._w > 0 and self._h > 0:
            painter = QPainter()
            painter.begin(self)

            painter.drawPixmap(0, 0, self.pixmap())
            painter.end()


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
        np_im = cv.convertScaleAbs(np_im, alpha=alpha, beta=beta)

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


class NNPreviewCanvas(MainCanvas):
    def __init__(self, parent: QWidget) -> None:
        super(NNPreviewCanvas, self).__init__(parent)

        self.pen_size = 5
        self.pen_color = QColor()
        self.points = []

    def draw_preview(self, parent_ann_layer: PaintingCanvas) -> None:
        input_pmap = parent_ann_layer.pixmap()
        self.pen_color = parent_ann_layer.pen_colour

        qim = input_pmap.toImage()
        im = q2n.rgb_view(qim)
        im_out = self.im_fill(im)
        self.points = self.outline_points(im_out)

        # Convert the binary mask to a semi-transparent overlay in the annotation layer's colour
        im_out = cv.cvtColor(im_out, cv.COLOR_GRAY2RGBA)
        fg_pixels = (im_out[:, :, 0:3] == [255, 255, 255]).all(2)
        bg_pixels = (im_out[:, :, 0:3] == [0, 0, 0]).all(2)
        im_out[fg_pixels] = (self.pen_color.red(), self.pen_color.green(), self.pen_color.blue(), 128)
        im_out[bg_pixels] = (0, 0, 0, 0)
        im_out = q2n.array2qimage(im_out)
        self.setPixmap(QPixmap.fromImage(im_out))

        self.update()

    @staticmethod
    def im_fill(im: np.ndarray) -> np.ndarray:
        # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
        im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        thresh, im_th = cv.threshold(im, 10, 255, cv.THRESH_BINARY)

        # Dilate/erode to get fill small gaps in lines
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        dilated = cv.dilate(im_th, element)
        eroded = cv.erode(dilated, element)
        im_th = eroded

        # Copy the thresholded image.
        im_floodfill = im_th.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Floodfill from point (0, 0)
        cv.floodFill(im_floodfill, mask, (0, 0), 255)
        # Invert floodfilled image
        im_floodfill_inv = cv.bitwise_not(im_floodfill)
        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv

        # Erode/dilate to get rid of any small specks
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        eroded = cv.erode(im_out, element)
        dilated = cv.dilate(eroded, element)
        im_out = dilated

        # Display images.
        # cv.imshow("Thresholded Image", im_th)
        # cv.imshow("Floodfilled Image", im_floodfill)
        # cv.imshow("Inverted Floodfilled Image", im_floodfill_inv)
        # cv.imshow("Foreground", im_out)
        # cv.imshow("Dilated", dilated)
        # cv.waitKey(0)

        return im_out

    @staticmethod
    def outline_points(im: np.ndarray) -> list:
        points = []
        # im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        _, contours, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # '[0][:-1])
        if len(contours) >= 1:
            for cnt in contours:
                epsilon = 0.001 * cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, epsilon, True)
                points.append(approx)
        # hull = cv.convexHull(cnt)

        # Display contours
        # result_borders = np.zeros(im.shape, np.uint8)
        # cv.drawContours(result_borders, contours, -1, 255, 1)
        # cv.drawContours(result_borders, approx, -1, 255, 3)
        # cv.imshow("Points on boundary", result_borders)
        # cv.waitKey(0)

        return points

    def paintEvent(self, event) -> None:
        if self._w > 0 and self._h > 0:
            painter = QPainter()
            painter.begin(self)

            p = painter.pen()
            p.setWidth(self.pen_size)
            p.setColor(self.pen_color)
            painter.setPen(p)
            painter.drawPixmap(0, 0, self.pixmap())

            if self.points:
                for group in self.points:
                    for point in group:
                        p = point[0]
                        p = QPoint(p[0], p[1])
                        painter.drawPoint(p)

            painter.end()
