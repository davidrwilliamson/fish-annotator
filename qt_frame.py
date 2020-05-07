from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtCore import QTimer
import pygame
import sys

# https://stackoverflow.com/questions/38280057/how-to-integrate-pygame-and-pyqt4
# https://stackoverflow.com/questions/52174384/pygame-refresh-in-pyqt
# https://github.com/mrexodia/pygame_qt/blob/master/game.py

class ImageWidget(QWidget):
    def __init__(self, surface, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.w = surface.get_width()
        self.h = surface.get_height()
        self.data = surface.get_buffer().raw
        self.image = QImage(self.data, self.w, self.h, QImage.Format_RGB32)

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        qp.drawImage(0, 0, self.image)
        qp.end()

    def update_surface(self, surface):
        self.data = surface.get_buffer().raw
        self.image = QImage(self.data, self.w, self.h, QImage.Format_RGB32)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        s = pygame.Surface((640, 480))
        s.fill((64, 128, 192, 224))
        pygame.draw.circle(s, (255, 255, 255, 255), (100, 100), 50)

        self.surface = s
        self.colour = (0, 0, 0)
        self.iw = ImageWidget(self.surface)
        self.setCentralWidget(self.iw)

        self.pygame_init()

    def pygame_init(self):
        pygame.init()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_pygame)
        self.timer.start(1000/30)

    def update_pygame(self):
        self.loop()
        self.iw.update_surface(self.surface)


    def loop(self):
        self.colour = (abs(self.colour[0] - 255), abs(self.colour[0] - 255), abs(self.colour[0] - 255))
        self.surface.fill(self.colour)


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec_()