from enum import IntEnum
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor, QPixmap, QPainter
from PyQt5.QtWidgets import QCheckBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget


class FrameToggle(IntEnum):
    BAD = 0
    INTERESTING = 1
    OTHER = 2


class ToolBtn(IntEnum):
    PAINT = 0
    ERASE = 1
    CLEAR = 2
    REVERT = 3
    ELLIPSE = 4


class NavBtn(IntEnum):
    START = 0
    END = 1
    PREV = 2
    NEXT = 3
    NOCHANGE = 4
    RANDOM = 5


class BottomButtons(QWidget):
    sgnl_change_frame = pyqtSignal(IntEnum)
    sgnl_adjust_brightness = pyqtSignal(int)
    sgnl_adjust_contrast = pyqtSignal(int)
    sgnl_toggle_scale_bar = pyqtSignal(bool)
    sgnl_toggle_zoom = pyqtSignal(bool)

    def __init__(self, parent: QWidget = None) -> None:
        super(BottomButtons, self).__init__(parent)
        bb_layout = QVBoxLayout(self)

        cb_layout = QHBoxLayout()

        self.cb_scale_bar = QCheckBox('Show scale bar')
        # cb_layout.addWidget(self.cb_scale_bar, alignment=Qt.AlignRight)
        self.cb_scale_bar.toggled.connect(self._call_toggle_scale_bar)
        self.cb_scale_bar.setChecked(False)

        self.cb_zoom = QCheckBox('2x &Zoom')
        cb_layout.addWidget(self.cb_zoom, alignment=Qt.AlignRight)
        self.cb_zoom.toggled.connect(self._call_toggle_zoom)
        self.cb_zoom.setChecked(False)
        # self.cb_zoom.setEnabled(False)

        bb_layout.addLayout(cb_layout)

        self.btn_start = QPushButton('Start (&{)')
        self.btn_end = QPushButton('End (&})')
        self.btn_prev = QPushButton('Previous (&[)')
        self.btn_next = QPushButton('Next (&])')
        self.btn_rand = QPushButton('Random (&?)')

        layout_buttons = QHBoxLayout()
        layout_buttons.addWidget(self.btn_start)
        layout_buttons.addWidget(self.btn_prev)
        layout_buttons.addWidget(self.btn_next)
        layout_buttons.addWidget(self.btn_end)
        layout_buttons.addWidget(self.btn_rand)

        self.btn_prev.clicked.connect(lambda: self._call_change(NavBtn.PREV))
        self.btn_next.clicked.connect(lambda: self._call_change(NavBtn.NEXT))
        self.btn_start.clicked.connect(lambda: self._call_change(NavBtn.START))
        self.btn_end.clicked.connect(lambda: self._call_change(NavBtn.END))
        self.btn_rand.clicked.connect(lambda: self._call_change(NavBtn.RANDOM))

        lbl_brightness = QLabel('Brightness')
        lbl_contrast = QLabel('Contrast')
        self._sldr_brightness = QSlider(Qt.Horizontal)
        self._sldr_brightness.setRange(-100, 100)
        self._sldr_brightness.setValue(0)
        self._sldr_contrast = QSlider(Qt.Horizontal)
        self._sldr_contrast.setRange(0, 40)
        self._sldr_contrast.setValue(10)

        layout_sliders = QGridLayout()
        layout_sliders.addWidget(lbl_brightness, 0, 0)
        layout_sliders.addWidget(lbl_contrast, 0, 1)
        layout_sliders.addWidget(self._sldr_brightness, 1, 0)
        layout_sliders.addWidget(self._sldr_contrast, 1, 1)

        bb_layout.addLayout(layout_sliders)
        bb_layout.addLayout(layout_buttons)

        self._sldr_brightness.valueChanged.connect(self._call_sldr_brightness)
        self._sldr_contrast.valueChanged.connect(self._call_sldr_contrast)

        self.enable_buttons(False)

    def _call_change(self, value: IntEnum) -> None:
        self.sgnl_change_frame.emit(value)

    def _call_sldr_brightness(self, value: int) -> None:
        self.sgnl_adjust_brightness.emit(value)

    def _call_sldr_contrast(self, value: int) -> None:
        self.sgnl_adjust_contrast.emit(value)

    def _call_toggle_scale_bar(self, checked: bool) -> None:
        self.sgnl_toggle_scale_bar.emit(checked)

    def _call_toggle_zoom(self, checked: bool) -> None:
        self.sgnl_toggle_zoom.emit(checked)

    def enable_buttons(self, enable: bool = True) -> None:
        """Sets all buttons to enabled (by default) or disable (if passed False as argument)."""
        buttons = [self.btn_start, self.btn_end, self.btn_next, self.btn_prev, self.btn_rand,
                   self._sldr_brightness, self._sldr_contrast, self.cb_zoom]
        for btn in buttons:
            btn.setEnabled(enable)


class RightButtons(QWidget):
    sgnl_change_im_layer = pyqtSignal(int)
    sgnl_change_ann_layer = pyqtSignal(bool, int)
    sgnl_toggle_rois = pyqtSignal(bool)
    sgnl_toggle_nn_preview = pyqtSignal(bool)
    sgnl_adjust_brush_size = pyqtSignal(int)
    sgnl_change_tool = pyqtSignal(IntEnum)

    def __init__(self, parent: QWidget = None) -> None:
        super(RightButtons, self).__init__(parent)
        layout = QVBoxLayout(self)
        layout_top_row = QHBoxLayout()

        layout_ann_layers = QVBoxLayout()
        layout_paint_tools = QVBoxLayout()
        layout_im_layers = QVBoxLayout()

        layout_top_row.addLayout(layout_ann_layers)
        layout_top_row.addLayout(layout_paint_tools)

        layout.addLayout(layout_top_row)
        layout.addLayout(layout_im_layers)

        # Annotation layers
        # Row 0, Column 0
        lbl_ann_layers = QLabel('Annotation layers')
        lbl_ann_layers.setAlignment(Qt.AlignCenter)
        self.btn_ann_0 = QPushButton('Myotome (&1)')
        self.btn_ann_1 = QPushButton('Eyes (&2)')
        self.btn_ann_2 = QPushButton('Yolk (&3)')
        self.btn_ann_3 = QPushButton('Embryo (&4)')
        self.btn_ann_4 = QPushButton('Egg (&5)')

        layout_ann_layers.addWidget(lbl_ann_layers, 1)
        layout_ann_layers.addWidget(self.btn_ann_0)
        layout_ann_layers.addWidget(self.btn_ann_1)
        layout_ann_layers.addWidget(self.btn_ann_2)
        layout_ann_layers.addWidget(self.btn_ann_3)
        layout_ann_layers.addWidget(self.btn_ann_4)

        for btn in [self.btn_ann_0, self.btn_ann_1, self.btn_ann_2, self.btn_ann_3, self.btn_ann_4]:
            btn.setCheckable(True)

        self.btn_ann_0.clicked.connect(lambda checked, idx=0: self._call_change_ann_layer(checked, idx))
        self.btn_ann_1.clicked.connect(lambda checked, idx=1: self._call_change_ann_layer(checked, idx))
        self.btn_ann_2.clicked.connect(lambda checked, idx=2: self._call_change_ann_layer(checked, idx))
        self.btn_ann_3.clicked.connect(lambda checked, idx=3: self._call_change_ann_layer(checked, idx))
        self.btn_ann_4.clicked.connect(lambda checked, idx=4: self._call_change_ann_layer(checked, idx))

        # Image layers
        # Row 1, Column 0
        lbl_im_layers = QLabel('Image layers')
        lbl_im_layers.setAlignment(Qt.AlignCenter)
        self.btn_raw_im = QPushButton('Raw Image')
        self.btn_bg_sub = QPushButton('Background Subtracted')
        self.btn_bg_im = QPushButton('Background')
        self.btn_bm_im = QPushButton('Binary Mask')
        self.btn_rois = QPushButton('RoIs')
        self.btn_nn_preview = QPushButton('&NN Input Preview')
        for btn in [self.btn_raw_im, self.btn_bg_sub, self.btn_bg_im, self.btn_bm_im, self.btn_rois, self.btn_nn_preview]:
            btn.setCheckable(True)

        self.btn_raw_im.clicked.connect(lambda: self._call_change_im_layer(0))
        self.btn_bg_im.clicked.connect(lambda: self._call_change_im_layer(1))
        self.btn_bm_im.clicked.connect(lambda: self._call_change_im_layer(2))
        self.btn_bg_sub.clicked.connect(lambda: self._call_change_im_layer(3))
        self.btn_rois.toggled.connect(self._call_toggle_rois)
        self.btn_nn_preview.toggled.connect(self._call_toggle_nn_preview)

        self.btns_im_layers = [self.btn_raw_im, self.btn_bg_im, self.btn_bm_im, self.btn_bg_sub]

        layout_im_layers.addWidget(lbl_im_layers, 1)
        layout_im_layers.addWidget(self.btn_raw_im)
        layout_im_layers.addWidget(self.btn_bg_im)
        layout_im_layers.addWidget(self.btn_bg_sub)
        layout_im_layers.addWidget(self.btn_bm_im)
        layout_im_layers.addWidget(self.btn_rois)
        layout_im_layers.addWidget(self.btn_nn_preview)

        # Paint tools
        # Row 0, Column 1
        lbl_paint = QLabel('Paint Tools')
        lbl_paint.setAlignment(Qt.AlignCenter)

        layout_curr_brush = QHBoxLayout()
        lbl_curr_brush_txt = QLabel('Brush: ')
        self.lbl_curr_brush_img = QLabel()
        im = QPixmap(20, 20)
        im.fill(QColor('transparent'))
        self.lbl_curr_brush_img.setPixmap(im)
        layout_curr_brush.addWidget(lbl_curr_brush_txt)
        layout_curr_brush.addWidget(self.lbl_curr_brush_img)

        self._sldr_brush_size = QSlider(Qt.Horizontal)
        self._sldr_brush_size.setRange(1, 20)
        self._sldr_brush_size.setValue(2)
        self._sldr_brush_size.valueChanged.connect(self._call_adjust_brush_size)

        self.btn_paint = QPushButton('Paint&brush')
        self.btn_erase = QPushButton('&Erase')
        self.btn_ellipse = QPushButton('E&llipse')
        self.btn_clear = QPushButton('&Clear')
        self.btn_revert = QPushButton('&Revert')
        self.btns_painting = [self.btn_paint, self.btn_erase, self.btn_ellipse]
        for btn in self.btns_painting:
            btn.setCheckable(True)

        layout_paint_tools.addWidget(lbl_paint, 1)
        layout_paint_tools.addLayout(layout_curr_brush)
        layout_paint_tools.addWidget(self._sldr_brush_size)
        layout_paint_tools.addWidget(self.btn_paint)
        layout_paint_tools.addWidget(self.btn_erase)
        layout_paint_tools.addWidget(self.btn_ellipse)
        layout_paint_tools.addWidget(self.btn_clear)
        layout_paint_tools.addWidget(self.btn_revert)

        self.btn_paint.clicked.connect(lambda: self._call_change_tool(ToolBtn.PAINT))
        self.btn_erase.clicked.connect(lambda: self._call_change_tool(ToolBtn.ERASE))
        self.btn_ellipse.clicked.connect(lambda: self._call_change_tool(ToolBtn.ELLIPSE))
        self.btn_clear.clicked.connect(lambda: self._call_change_tool(ToolBtn.CLEAR))
        self.btn_revert.clicked.connect(lambda: self._call_change_tool(ToolBtn.REVERT))

        self.enable_buttons(False)

    def _call_change_im_layer(self, idx: int) -> None:
        self.sgnl_change_im_layer.emit(idx)
        buttons = self.btns_im_layers
        self.uncheck_others(buttons, idx)

    def _call_change_ann_layer(self, checked: bool, idx: int) -> None:
        self.sgnl_change_ann_layer.emit(checked, idx)

        buttons = [self.btn_ann_0, self.btn_ann_1, self.btn_ann_2, self.btn_ann_3, self.btn_ann_4]
        for i in range(len(buttons)):
            if i != idx:
                buttons[i].setChecked(False)

    def _call_toggle_rois(self, checked: bool) -> None:
        self.sgnl_toggle_rois.emit(checked)

    def _call_toggle_nn_preview(self, checked: bool) -> None:
        self.sgnl_toggle_nn_preview.emit(checked)

    def _call_adjust_brush_size(self, value: int) -> None:
        self.sgnl_adjust_brush_size.emit(value)

    def _call_change_tool(self, idx: IntEnum) -> None:
        self.sgnl_change_tool.emit(idx)

    @staticmethod
    def uncheck_others(buttons: list, btn: int) -> None:
        for i in range(len(buttons)):
            if i == btn:
                buttons[i].setChecked(True)
            else:
                buttons[i].setChecked(False)

    def enable_buttons(self, enable: bool = True, selection=range(16)) -> None:
        """Sets all buttons to enabled (by default) or disable (if passed False as argument)."""
        buttons = [self.btn_ann_0, self.btn_ann_1, self.btn_ann_2, self.btn_ann_3, self.btn_ann_4,
                   self.btn_raw_im, self.btn_bg_im, self.btn_bm_im, self.btn_bg_sub, self.btn_rois, self.btn_nn_preview,
                   self.btn_paint, self.btn_erase, self.btn_ellipse, self.btn_clear, self.btn_revert,
                   self._sldr_brush_size]
        for btn in selection:
            buttons[btn].setEnabled(enable)

    def set_lbl_curr_brush(self, ann_canvas, draw: bool) -> None:
        im = self.lbl_curr_brush_img.pixmap()
        painter = QPainter(im)

        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        extents = im.rect()
        painter.eraseRect(extents)

        if draw:
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            colour = ann_canvas.pen_colour
            size = ann_canvas.pen_size
            p = painter.pen()
            p.setColor(colour)
            p.setWidth(1)
            painter.setPen(p)
            painter.setBrush(colour)
            painter.drawRect(size / 2, size / 2, size, size)

        painter.end()
        self.lbl_curr_brush_img.update()


class BottomRightButtons(QWidget):
    # We want check boxes for: bad frame, frame of interest
    # Maybe also for more than one fish in frame? Text box that takes a number?
    sgnl_toggle_bad_frame = pyqtSignal(bool)
    sgnl_toggle_interesting_frame = pyqtSignal(bool)

    def __init__(self, parent: QWidget = None) -> None:
        super(BottomRightButtons, self).__init__(parent)
        brb_layout = QGridLayout(self)

        self.cb_bad = QCheckBox('Bad frame')
        self.cb_interest = QCheckBox('Interesting frame')
        self.cb_bad.setTristate(False)
        self.cb_interest.setTristate(False)

        brb_layout.addWidget(self.cb_bad, 0, 0)
        brb_layout.addWidget(self.cb_interest, 1, 0)

        self.cb_bad.toggled.connect(self._call_toggle_bad_frame)
        self.cb_interest.toggled.connect(self._call_toggle_interesting_frame)

        self.enable_buttons(False)

    def _call_toggle_bad_frame(self, checked: bool) -> None:
        self.sgnl_toggle_bad_frame.emit(checked)

    def _call_toggle_interesting_frame(self, checked: bool) -> None:
        self.sgnl_toggle_interesting_frame.emit(checked)

    def enable_buttons(self, enable: bool = True) -> None:
        """Sets all buttons to enabled (by default) or disable (if passed False as argument)."""
        buttons = [self.cb_bad, self.cb_interest]
        for btn in buttons:
            btn.setEnabled(enable)


class LeftButtons(QWidget):
    # TODO: Add markers showing which layers exist for the current frame
    # TODO: Add counters giving how many frames have annotations
    sgnl_toggle_bad_frames = pyqtSignal(bool)
    sgnl_toggle_interesting_frames = pyqtSignal(bool)
    sgnl_toggle_other_frames = pyqtSignal(bool)

    def __init__(self, parent: QWidget = None) -> None:
        super(LeftButtons, self).__init__(parent)
        lb_layout = QVBoxLayout(self)

        self.lbl_frames = QLabel('\n')
        self.lbl_interesting_frames = QLabel('\n')
        self.lbl_bad_frames = QLabel('\n')
        self.lbl_annotations = QLabel('\n')

        self.cb_bad = QCheckBox('Show bad frames')
        self.cb_interest = QCheckBox('Show interesting frames')
        self.cb_other = QCheckBox('Show other frames')

        self.cb_bad.setTristate(False)
        self.cb_interest.setTristate(False)
        self.cb_other.setTristate(False)
        self.cb_bad.setEnabled(False)
        self.cb_interest.setEnabled(False)
        self.cb_other.setEnabled(False)

        lb_layout.addWidget(self.lbl_frames)
        lb_layout.addWidget(self.lbl_interesting_frames)
        lb_layout.addWidget(self.lbl_bad_frames)
        lb_layout.addWidget(self.lbl_annotations)

        lb_layout.addWidget(self.cb_interest)
        lb_layout.addWidget(self.cb_other)
        lb_layout.addWidget(self.cb_bad)

        self.cb_bad.toggled.connect(lambda checked: self._call_toggled(checked, FrameToggle.BAD))
        self.cb_interest.toggled.connect(lambda checked: self._call_toggled(checked, FrameToggle.INTERESTING))
        self.cb_other.toggled.connect(lambda checked: self._call_toggled(checked, FrameToggle.OTHER))

    def _call_toggled(self, checked: bool, option: IntEnum) -> None:
        if option == FrameToggle.BAD:
            self.sgnl_toggle_bad_frames.emit(checked)
        if option == FrameToggle.INTERESTING:
            self.sgnl_toggle_interesting_frames.emit(checked)
        if option == FrameToggle.OTHER:
            self.sgnl_toggle_other_frames.emit(checked)

    def update_labels(self, num_frames, cf_no, i_f, b_f, ann) -> None:
        self.lbl_frames.setText('Frame: {} / {}\n'.format(cf_no, num_frames))
        self.lbl_interesting_frames.setText('Interesting frames: {}\n'.format(i_f))
        self.lbl_bad_frames.setText('Bad frames: {}\n'.format(b_f))
        self.lbl_bad_frames.setText('Annotations:\n  Myotome {}\n  Eyes  {}\n  Yolk  {}\n  Embyro  {}\n  Egg  {}'
                                    .format(len(ann[0]), len(ann[1]), len(ann[2]), len(ann[3]), len(ann[4])))

    def enable_buttons(self, show_bad: bool, show_other: bool, show_interesting: bool) -> None:
        self.cb_bad.setEnabled(True)
        self.cb_bad.setChecked(show_bad)

        self.cb_other.setEnabled(True)
        self.cb_other.setChecked(show_other)

        self.cb_interest.setEnabled(True)
        self.cb_interest.setChecked(show_interesting)
