from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QCheckBox, QGridLayout, QLabel, QPushButton, QWidget


class BottomButtons(QWidget):
    sgnl_change_frame = pyqtSignal(int)

    def __init__(self, parent: QWidget = None) -> None:
        super(BottomButtons, self).__init__(parent)
        bb_layout = QGridLayout(self)

        self.btn_play = QPushButton('Play')
        self.btn_pause = QPushButton('Pause')
        self.btn_prev = QPushButton('Previous')
        self.btn_next = QPushButton('Next')

        bb_layout.addWidget(self.btn_play, 0, 0)
        bb_layout.addWidget(self.btn_pause, 0, 1)
        bb_layout.addWidget(self.btn_prev, 1, 0)
        bb_layout.addWidget(self.btn_next, 1, 1)

        self.btn_prev.clicked.connect(lambda: self.call_change(-1))
        self.btn_next.clicked.connect(lambda: self.call_change(1))

        self.enable_buttons(False)

    def call_change(self, direction: int) -> None:
        self.sgnl_change_frame.emit(direction)

    def enable_buttons(self, enable: bool = True) -> None:
        """Sets all buttons to enabled (by default) or disable (if passed False as argument)."""
        buttons = [self.btn_play, self.btn_pause, self.btn_next, self.btn_prev]
        for btn in buttons:
            btn.setEnabled(enable)


class RightButtons(QWidget):
    sgnl_change_im_layer = pyqtSignal(int)
    sgnl_toggle_rois = pyqtSignal(bool)

    def __init__(self, parent: QWidget = None) -> None:
        super(RightButtons, self).__init__(parent)
        rb_layout = QGridLayout(self)

        # Column 0
        lbl_layers = QLabel('Layers')
        lbl_layers.setAlignment(Qt.AlignCenter)
        self.btn_raw_im = QPushButton('Raw Image')
        self.btn_bg_sub = QPushButton('Background Subtracted')
        self.btn_bg_im = QPushButton('Background')
        self.btn_bm_im = QPushButton('Binary Mask')
        self.btn_rois = QPushButton('RoIs')
        for btn in [self.btn_raw_im, self.btn_bg_sub, self.btn_bg_im, self.btn_bm_im, self.btn_rois]:
            btn.setCheckable(True)

        self.btn_raw_im.clicked.connect(lambda: self.call_btn(0))
        self.btn_bg_im.clicked.connect(lambda: self.call_btn(1))
        self.btn_bm_im.clicked.connect(lambda: self.call_btn(2))
        self.btn_bg_sub.clicked.connect(lambda: self.call_btn(3))
        self.btn_rois.toggled.connect(self.call_rois)

        rb_layout.addWidget(lbl_layers, 0, 0)
        rb_layout.addWidget(self.btn_raw_im, 1, 0)
        rb_layout.addWidget(self.btn_bg_im, 3, 0)
        rb_layout.addWidget(self.btn_bg_sub, 2, 0)
        rb_layout.addWidget(self.btn_bm_im, 4, 0)
        rb_layout.addWidget(self.btn_rois, 5, 0)

        # Column 1
        lbl_paint = QLabel('Paint Tools')
        lbl_paint.setAlignment(Qt.AlignCenter)
        self.btn_paint = QPushButton('Paintbrush')
        self.btn_fill = QPushButton('Fill')
        self.btn_erase = QPushButton('Erase')
        self.btn_erase.setCheckable(True)

        rb_layout.addWidget(lbl_paint, 0, 1)
        rb_layout.addWidget(self.btn_paint, 1, 1)
        rb_layout.addWidget(self.btn_fill, 2, 1)
        rb_layout.addWidget(self.btn_erase, 3, 1)

        self.enable_buttons(False)

    def call_btn(self, idx: int) -> None:
        self.sgnl_change_im_layer.emit(idx)
        self.uncheck_others(idx)

    def call_rois(self, checked: bool) -> None:
        self.sgnl_toggle_rois.emit(checked)

    def uncheck_others(self, btn: int) -> None:
        buttons = [self.btn_raw_im, self.btn_bg_im, self.btn_bm_im, self.btn_bg_sub]
        for i in range(len(buttons)):
            if i != btn:
                buttons[i].setChecked(False)

    def enable_buttons(self, enable: bool = True) -> None:
        """Sets all buttons to enabled (by default) or disable (if passed False as argument)."""
        buttons = [self.btn_raw_im, self.btn_bg_im, self.btn_bm_im, self.btn_bg_sub, self.btn_rois,
                   self.btn_paint, self.btn_fill, self.btn_erase]
        for btn in buttons:
            btn.setEnabled(enable)


class BottomRightButtons(QWidget):
    # We want check boxes for: bad frame, frame of interest
    # Maybe also for more than one fish in frame? Text box that takes a number?
    sgnl_cb_bad_changed = pyqtSignal(int)
    sgnl_cb_interest_changed = pyqtSignal(int)

    def __init__(self, parent: QWidget = None) -> None:
        super(BottomRightButtons, self).__init__(parent)
        brb_layout = QGridLayout(self)

        self.cb_bad = QCheckBox('Bad frame')
        self.cb_interest = QCheckBox('Interesting frame')

        brb_layout.addWidget(self.cb_bad, 0, 0)
        brb_layout.addWidget(self.cb_interest, 1, 0)

        self.cb_bad.stateChanged.connect(self.call_cb_bad)
        self.cb_interest.stateChanged.connect(self.call_cb_interest)

        self.enable_buttons(False)

    def call_cb_bad(self, state: int) -> None:
        self.sgnl_cb_bad_changed.emit(state)

    def call_cb_interest(self, state: int) -> None:
        self.sgnl_cb_interest_changed.emit(state)

    def enable_buttons(self, enable: bool = True) -> None:
        """Sets all buttons to enabled (by default) or disable (if passed False as argument)."""
        buttons = [self.cb_bad, self.cb_interest]
        for btn in buttons:
            btn.setEnabled(enable)