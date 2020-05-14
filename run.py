from PyQt5 import QtWidgets
from main_window import MainWindow


def run() -> None:
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    run()
