from PyQt5.QtCore import pyqtSlot, QBuffer, QByteArray, QIODevice, QTimer
from PyQt5.QtGui import QKeyEvent, QPen
from PyQt5.QtWidgets import QGridLayout, QLabel, QMainWindow, QWidget
from typing import List

from buttons import *
from canvases import *
from menus import *
from analysis import *


class MainWindow(QMainWindow):
    def __init__(self, parent: QMainWindow = None) -> None:
        super(MainWindow, self).__init__(parent)

        # self.popup = None

        self.im_folder: ImageFolder = None
        self.curr_layer: int = 0
        self.curr_ann_layer: int = -1
        self.draw_rois: bool = False

        # Menu bar
        self.main_menu = MainMenu(self)
        self.setMenuWidget(self.main_menu)

        # TODO: Add a border around the image, even when it isn't visible
        self.image_frame = ImageFrame(self)
        self.rois_canvas = RoIsCanvas(self)
        self.nn_preview_canvas = NNPreviewCanvas(self)
        self.lbl_frame = QLabel('\n')
        # self.lbl_frame_no = QLabel('\n')
        # self.lbl_frame_no.setAlignment(Qt.AlignRight)

        self.scale_bar = ScaleBar(self.image_frame, 1)
        self.scale_bar.setVisible(False)

        self.saved_canvases: List[QBuffer] = []
        self.annotation_canvases: List[PaintingCanvas] = []
        self.annotation_canvases.append(PaintingCanvas(self, 'red'))
        self.annotation_canvases.append(PaintingCanvas(self, 'blue'))
        self.annotation_canvases.append(PaintingCanvas(self, 'green'))
        self.annotation_canvases.append(PaintingCanvas(self, 'purple'))
        self.annotation_canvases.append(PaintingCanvas(self, 'cyan'))

        self.change_ann_layer(False, self.curr_ann_layer)  # Disables and hides annotation canvases

        # Set up layout of window inside central placeholder widget
        placeholder = QWidget()
        self.setCentralWidget(placeholder)
        grid_layout = QGridLayout(placeholder)
        grid_layout.addWidget(self.lbl_frame, 0, 1)
        # grid_layout.addWidget(self.lbl_frame_no, 0, 1)
        grid_layout.addWidget(self.image_frame, 1, 1)
        grid_layout.addWidget(self.rois_canvas, 1, 1)
        grid_layout.addWidget(self.nn_preview_canvas, 1, 1)
        for canvas in self.annotation_canvases:
            grid_layout.addWidget(canvas, 1, 1)

        # Add various sets of buttons
        self.bottom_buttons = BottomButtons()
        grid_layout.addWidget(self.bottom_buttons, 2, 1)
        self.right_buttons = RightButtons()
        grid_layout.addWidget(self.right_buttons, 1, 2)
        self.br_buttons = BottomRightButtons()
        grid_layout.addWidget(self.br_buttons, 2, 2)
        self.left_buttons = LeftButtons()
        grid_layout.addWidget(self.left_buttons, 1, 0)

        # Connect up button signals
        self.main_menu.sgnl_im_folder.connect(self.set_im_folder)
        self.main_menu.sgnl_export_menu.connect(self.export_menu)
        self.main_menu.sgnl_analysis_menu.connect(self.analysis_menu)
        self.main_menu.sgnl_save_ann.connect(self.save_ann)
        self.right_buttons.sgnl_change_im_layer.connect(self.change_im_layer)
        self.right_buttons.sgnl_change_ann_layer.connect(self.change_ann_layer)
        self.right_buttons.sgnl_change_tool.connect(self.change_tool)
        self.right_buttons.sgnl_toggle_rois.connect(self.toggle_rois)
        self.right_buttons.sgnl_toggle_nn_preview.connect(self.toggle_nn_preview)
        self.right_buttons.sgnl_adjust_brush_size.connect(self.adjust_brush_size)
        self.bottom_buttons.sgnl_change_frame.connect(self.change_frame)
        self.bottom_buttons.sgnl_adjust_brightness.connect(self.adjust_brightness)
        self.bottom_buttons.sgnl_adjust_contrast.connect(self.adjust_contrast)
        self.bottom_buttons.sgnl_toggle_scale_bar.connect(self.toggle_scale_bar)
        self.bottom_buttons.sgnl_toggle_zoom.connect(self.toggle_zoom)
        self.br_buttons.sgnl_toggle_bad_frame.connect(self.toggle_bad_frame)
        self.br_buttons.sgnl_toggle_interesting_frame.connect(self.toggle_interesting_frame)
        self.left_buttons.sgnl_toggle_bad_frames.connect(self.show_bad_frames)
        self.left_buttons.sgnl_toggle_interesting_frames.connect(self.show_interesting_frames)
        self.left_buttons.sgnl_toggle_other_frames.connect(self.show_other_frames)

        self.setWindowTitle("Fish Annotator")

    def keyPressEvent(self, e: QKeyEvent) -> None:
        k_p = e.key()
        # Only if paint button is already enabled, meaning we're in annotation mode
        if self.right_buttons.btn_paint.isEnabled():
            if k_p == Qt.Key_B:
                self.change_tool(ToolBtn.PAINT)
            elif k_p == Qt.Key_E:
                self.change_tool(ToolBtn.ERASE)
            elif k_p == Qt.Key_L:
                self.change_tool(ToolBtn.ELLIPSE)
            if k_p == Qt.Key_N:
                self.right_buttons.btn_nn_preview.click()
            if k_p == Qt.Key_C:
                self.right_buttons.btn_clear.click()
            if k_p == Qt.Key_R:
                self.right_buttons.btn_revert.click()

        if self.im_folder:
            if k_p == Qt.Key_BracketLeft:
                self.change_frame(NavBtn.PREV)
            elif k_p == Qt.Key_BracketRight:
                self.change_frame(NavBtn.NEXT)
            if k_p == Qt.Key_BraceLeft:
                self.change_frame(NavBtn.START)
            if k_p == Qt.Key_BraceRight:
                self.change_frame(NavBtn.END)
            if k_p == Qt.Key_Question:
                self.change_frame(NavBtn.RANDOM)
            if k_p == Qt.Key_1:
                self.right_buttons.btn_ann_0.click()
            if k_p == Qt.Key_2:
                self.right_buttons.btn_ann_1.click()
            if k_p == Qt.Key_3:
                self.right_buttons.btn_ann_2.click()
            if k_p == Qt.Key_4:
                self.right_buttons.btn_ann_3.click()
            if k_p == Qt.Key_5:
                self.right_buttons.btn_ann_4.click()
            if k_p == Qt.Key_I:
                checked = self.br_buttons.cb_interest.isChecked()
                self.toggle_interesting_frame(not checked)
            if k_p == Qt.Key_Z:
                zoom_cb = self.bottom_buttons.cb_zoom
                checked = zoom_cb.isChecked()
                zoom_cb.setChecked(not checked)

    def closeEvent(self, event) -> None:
        self.save_annotations_mem()
        self.save_annotations_disk()

    def update_lbl_frame(self) -> None:
        folder = self.im_folder.folder
        cf_no, cf_fn = self.im_folder.framepos
        num_frames = self.im_folder.num_frames
        self.lbl_frame.setText('Folder:  {}\nFile:      {}'.format(folder, cf_fn))
        # self.lbl_frame_no.setText('\nFrame: {} / {}'.format(cf_no, num_frames))

        # TODO: Expose non-protected properties in ImageFolder for this, and/or find a nicer way to update these labels
        i_f = len(self.im_folder._interesting_frames)
        b_f = len(self.im_folder._bad_frames)
        ann = self.im_folder.annotations
        self.left_buttons.update_labels(num_frames, cf_no, i_f, b_f, ann)

    def save_annotations_mem(self) -> None:
        # The canvas saved to memory is always the full-sized annotation layer
        canvases = [None] * len(self.annotation_canvases)
        save: bool = False
        i = 0
        for canvas in self.annotation_canvases:
            if canvas.is_used:  # or canvas.is_cleared:  # Don't bother to save unless something has actually been drawn
                buffer = QBuffer()
                buffer.open(QIODevice.ReadWrite)
                canvas.pixmap_scaled(self.im_folder.frame_w, self.im_folder.frame_h).save(buffer, 'png')
                buffer.close()
                canvases[i] = buffer
                canvas.erase_all()
                canvas.is_cleared = False
                save = True
            i += 1
        if save:
            idx = self.im_folder.framepos[0]
            self.saved_canvases[idx] = canvases
            # Update im_folder.annotations with the indices of saved canvases
            k = 0
            for j in self.saved_canvases[idx]:
                if j is not None:
                    self.im_folder.annotations[k].append(idx)
                    self.im_folder.annotations[k] = np.unique(self.im_folder.annotations[k]).tolist()
                k += 1
            self.update_lbl_frame()

    def load_annotations_mem(self) -> None:
        k = self.im_folder.framepos[0]
        # If the current frame has ANY annotations on it, on any layer...
        if self.saved_canvases[k]:
            for i in range(len(self.saved_canvases[k])):
                # If this specific layer has an annotation on it, display that annotation
                if self.saved_canvases[k][i]:
                    buffer = self.saved_canvases[k][i].buffer()
                    self.annotation_canvases[i].pixmap().loadFromData(buffer, 'png')
                    self.annotation_canvases[i].is_used = True
                # Otherwise clear the layer so we aren't showing annotations from other frames
                else:
                    self.annotation_canvases[i].erase_all()
                self.annotation_canvases[i].update()
        # Clear all layers if this frame doesn't have any annotations
        else:
            for canvas in self.annotation_canvases:
                canvas.erase_all()
        # TODO: When we load the full-sized annotation from memory for the first time, we may need to scale them down for display

    def save_annotations_disk(self) -> None:
        if self.im_folder is None:
            return

        save_folder = os.path.join(self.im_folder.folder, 'analysis/annotations')
        try:
            os.makedirs(save_folder)
        except FileExistsError:
            pass

        for i in range(len(self.saved_canvases)):
            frame = self.saved_canvases[i]
            if frame:
                for j in range(len(frame)):
                    canvas = frame[j]
                    if canvas:
                        buffer: QByteArray = canvas.buffer()
                        pmap = QPixmap()
                        pmap.loadFromData(buffer, 'png')

                        save_path = os.path.join(save_folder, '{}_{}_{}.png'.format(self.im_folder._all_files[i], i, j))
                        pmap.save(save_path, 'png')

    def load_annotations_disk(self) -> None:
        # Because we save the annotations at full size, when loaded from disk they are also at full size
        save_folder = os.path.join(self.im_folder.folder, 'analysis/annotations')
        if os.path.isdir(save_folder):
            annotations = [file for file in os.listdir(save_folder) if os.path.splitext(file)[1] == '.png']
            for file in annotations:
                underscore_split = os.path.splitext(file)[0].split('_')
                if len(underscore_split) == 4:
                    i, j = list(map(int, os.path.splitext(file)[0].split('_')[2:]))
                elif len(underscore_split) == 3:
                    i, j = list(map(int, os.path.splitext(file)[0].split('_')[1:]))
                else:
                    raise
                full_path = os.path.join(save_folder, file)

                pmap = QPixmap()
                pmap.load(full_path, 'png')
                buffer = QBuffer()
                buffer.open(QIODevice.ReadWrite)
                is_saved = pmap.save(buffer, 'png')
                if not is_saved:
                    raise
                buffer.close()

                if self.saved_canvases[i] is None:
                    self.saved_canvases[i] = []
                if len(self.saved_canvases[i]) < j + 1:
                    for _ in range(j - len(self.saved_canvases[i]) + 1):
                        self.saved_canvases[i].append(None)
                self.saved_canvases[i][j] = buffer

    def adjust_frame_sizes(self, w, h) -> None:
        canvases = [self.image_frame, self.rois_canvas, self.nn_preview_canvas]
        canvases.extend(self.annotation_canvases)
        for canvas in canvases:
            canvas.set_frame_size(w, h)

    @pyqtSlot()
    def save_ann(self) -> None:
        self.save_annotations_mem()
        self.save_annotations_disk()
        self.load_annotations_disk()
        self.load_annotations_mem()

    @pyqtSlot(ImageFolder)
    def set_im_folder(self, im_folder: ImageFolder) -> None:
        # TODO: Reset inactive buttons/step out of annotation mode when changing image folders
        self.save_annotations_mem()
        self.save_annotations_disk()

        self.im_folder = im_folder
        self.adjust_frame_sizes(im_folder.frame_w, im_folder.frame_h)
        self.saved_canvases = [None for i in range(im_folder.num_frames)]
        self.load_annotations_disk()
        self.load_annotations_mem()
        self.change_im_layer(0)
        self.change_frame(NavBtn.NOCHANGE)
        self.right_buttons.enable_buttons(selection=range(10))
        self.right_buttons.uncheck_others(self.right_buttons.btns_im_layers, 0)
        self.bottom_buttons.enable_buttons()
        self.br_buttons.enable_buttons()
        self.left_buttons.enable_buttons(self.im_folder._show_bad, self.im_folder._show_other,
                                         self.im_folder._show_interesting)
        self.main_menu.enable_export()

    @pyqtSlot(int)
    def change_im_layer(self, im_idx: int) -> None:
        curr_frames = self.im_folder.curr_frames
        if curr_frames[im_idx]:  # Some frames might not have any layers associated with them other than im_raw
            im = curr_frames[im_idx]
            self.curr_layer = im_idx
            self.image_frame.update_image(im)
            self.update_lbl_frame()
        else:
            self.change_im_layer(0)  # In this case we switch back to im_raw
            # TODO: Disable buttons that don't have a corresponding layer available (including RoIs)

    @pyqtSlot(IntEnum)
    def change_tool(self, idx: IntEnum) -> None:
        ann_layer: PaintingCanvas = self.annotation_canvases[self.curr_ann_layer]
        if idx == ToolBtn.PAINT:
            ann_layer.brush_erase = False
            self.right_buttons.btn_paint.setChecked(True)
            self.right_buttons.btn_erase.setChecked(False)
            self.right_buttons.btn_ellipse.setChecked(False)
            ann_layer.toggle_ellipse_drawing(False)
        if idx == ToolBtn.ERASE:
            ann_layer.brush_erase = True
            self.right_buttons.btn_paint.setChecked(False)
            self.right_buttons.btn_erase.setChecked(True)
            self.right_buttons.btn_ellipse.setChecked(False)
            ann_layer.toggle_ellipse_drawing(False)
        if idx == ToolBtn.CLEAR:
            ann_layer.erase_all()
            # Had a bug where if a canvas was cleared and then we changed layers without drawing anything
            # it would be marked as unused and not saved.
            # Now we check for is_cleared when saving too, and set it to false after a save.
            # TODO: We do end up saving blank annotations in this case, though. Should fix.
            # ann_layer.is_cleared = True
            # My "fix" ended up introducing a new bug where annotations are not saved, because we call the clear tool in a few other places.
            # I guess we need a way to know that a previously not empty annotation has been made empty.
            ann_layer.update()
            self.change_tool(ToolBtn.PAINT)
        if idx == ToolBtn.REVERT:
            self.load_annotations_mem()
            ann_layer.update()
        if idx == ToolBtn.ELLIPSE:
            ann_layer.brush_erase = False
            self.right_buttons.btn_paint.setChecked(False)
            self.right_buttons.btn_erase.setChecked(False)
            checked = self.right_buttons.btn_ellipse.isChecked()
            ann_layer.toggle_ellipse_drawing(checked)

    @pyqtSlot(bool, int)
    def change_ann_layer(self, checked: bool, ann_idx: int) -> None:
        for canvas in self.annotation_canvases:
            canvas.setEnabled(False)
            canvas.setVisible(False)
            # self.right_buttons.enable_buttons(False, range(9, 11))
        if ann_idx >= 0:  # This means we won't try to set the brush image before a canvas exists
            self.save_annotations_mem()
            self.load_annotations_mem()

            if self.right_buttons.btn_nn_preview.isChecked():
                self.nn_preview_canvas.draw_preview(self.annotation_canvases[ann_idx])

            self.right_buttons.uncheck_others(self.right_buttons.btns_painting, -1)
            self.right_buttons.enable_buttons(False, selection=range(10, 16))
            self.curr_ann_layer = ann_idx
            self.right_buttons.set_lbl_curr_brush(self.annotation_canvases[ann_idx], checked)
            if checked:  # Lets us hide annotations again by unchecking button
                self.right_buttons.btn_paint.setChecked(True)
                self.change_tool(ToolBtn.PAINT)
                self.annotation_canvases[ann_idx].setEnabled(True)
                self.annotation_canvases[ann_idx].setVisible(True)
                self.right_buttons.enable_buttons(selection=range(10, 16))
            else:
                self.curr_ann_layer = -1

    @pyqtSlot(bool)
    def toggle_rois(self, checked: bool) -> None:
        self.draw_rois = checked
        if checked:
            rois = self.im_folder.rois
            self.rois_canvas.draw_rois(rois)
        else:
            self.rois_canvas.erase_rois()

    @pyqtSlot(bool)
    def toggle_nn_preview(self, checked: bool) -> None:
        if checked and self.curr_ann_layer >= 0:
            self.annotation_canvases[self.curr_ann_layer].setVisible(False)
            self.nn_preview_canvas.draw_preview(self.annotation_canvases[self.curr_ann_layer])
            self.nn_preview_canvas.setVisible(True)
        else:
            self.annotation_canvases[self.curr_ann_layer].setVisible(True)
            self.nn_preview_canvas.setVisible(False)
            self.right_buttons.btn_nn_preview.setChecked(False)

    @pyqtSlot(IntEnum)
    def change_frame(self, value: IntEnum) -> None:
        self.save_annotations_mem()
        if value is NavBtn.NEXT:
            self.im_folder.next_frame()
        elif value is NavBtn.PREV:
            self.im_folder.prev_frame()
        elif value is NavBtn.START:
            self.im_folder.go_to_first_frame()
        elif value is NavBtn.END:
            self.im_folder.go_to_frame(self.im_folder.last_frame)
        elif value is NavBtn.RANDOM:
            self.im_folder.random_frame()

        self.change_im_layer(self.curr_layer)
        if self.draw_rois:
            self.rois_canvas.draw_rois(self.im_folder.rois)
        self.load_annotations_mem()

        # When we change frames, update NN input preview
        if self.right_buttons.btn_nn_preview.isChecked():
            self.nn_preview_canvas.draw_preview(self.annotation_canvases[self.curr_ann_layer])

        # This is a gross hack that should be fixed/removed,
        # but I want to make sure that boxes are checked correctly when a frame is loaded
        if self.im_folder.framepos[0] in self.im_folder._interesting_frames:
            self.br_buttons.cb_interest.setChecked(True)
        else:
            self.br_buttons.cb_interest.setChecked(False)
        if self.im_folder.framepos[0] in self.im_folder._bad_frames:
            self.br_buttons.cb_bad.setChecked(True)
        else:
            self.br_buttons.cb_bad.setChecked(False)

    @pyqtSlot(int)
    def adjust_brightness(self, value: int) -> None:
        self.image_frame.set_brightness(value)

    @pyqtSlot(int)
    def adjust_contrast(self, value: int) -> None:
        self.image_frame.set_contrast(value)

    @pyqtSlot(int)
    def adjust_brush_size(self, value: int) -> None:
        for canvas in self.annotation_canvases:
            canvas.pen_size = value
        self.right_buttons.set_lbl_curr_brush(self.annotation_canvases[self.curr_ann_layer], True)

    @pyqtSlot(bool)
    def toggle_bad_frame(self, checked: bool) -> None:
        self.im_folder.toggle_bad_frame(checked)
        # A bad frame cannot be an interesting frame
        if checked:
            self.toggle_interesting_frame(False)
        self.change_frame(NavBtn.NOCHANGE)

    @pyqtSlot(bool)
    def toggle_interesting_frame(self, checked: bool) -> None:
        self.im_folder.toggle_interesting_frame(checked)
        # An interesting frame cannot be a bad frame
        if checked:
            self.toggle_bad_frame(False)
        self.change_frame(NavBtn.NOCHANGE)

    @pyqtSlot(bool)
    def toggle_scale_bar(self, checked: bool) -> None:
        if checked:
            self.scale_bar.setVisible(True)
        else:
            self.scale_bar.setVisible(False)

    @pyqtSlot(bool)
    def toggle_zoom(self, checked: bool) -> None:
        w, h = self.im_folder.frame_w, self.im_folder.frame_h
        if checked:
            w *= 2
            h *= 2

        canvases = [self.image_frame, self.rois_canvas, self.nn_preview_canvas]
        canvases.extend(self.annotation_canvases)
        for canvas in canvases:
            canvas.set_frame_size(w, h)
        self.change_frame(NavBtn.NOCHANGE)
        self.adjust_frame_sizes(w, h)

    @pyqtSlot(bool)
    def show_bad_frames(self, checked: bool) -> None:
        self.im_folder.toggle_show_bad(checked)
        self.change_frame(NavBtn.NOCHANGE)

    @pyqtSlot(bool)
    def show_interesting_frames(self, checked: bool) -> None:
        self.im_folder.toggle_show_interesting(checked)
        self.change_frame(NavBtn.NOCHANGE)

    @pyqtSlot(bool)
    def show_other_frames(self, checked: bool) -> None:
        self.im_folder.toggle_show_other(checked)
        self.change_frame(NavBtn.NOCHANGE)

    @pyqtSlot(IntEnum)
    def export_menu(self, option: IntEnum) -> None:
        # TODO: Export methods should run in their own thread so as not to block the GUI,
        #  with progress bar & notifications
        if option == ExportMenu.PREVIEW_ROIS:
            pass
            # self.popup = self.main_menu.preview_rois(self.im_folder)
            # updates = QTimer(self.popup)
            # updates.timeout.connect(lambda: self.main_menu.update_preview(self.im_folder, self.popup))
            # updates.start(200)
        elif option == ExportMenu.EXPORT_FULL:
            self.main_menu.export_full_frames(self.im_folder)
        elif option == ExportMenu.EXPORT_ROIS:
            self.main_menu.export_rois(self.im_folder)
        elif option == ExportMenu.EXPORT_MONTAGE:
            self.main_menu.export_montage(self.im_folder)
        elif option == ExportMenu.EXPORT_INTERESTING:
            self.main_menu.export_interesting(self.im_folder)
        elif option == ExportMenu.EXPORT_CURRENT:
            self.main_menu.export_current(self.im_folder, self.image_frame, self.rois_canvas,
                                          self.annotation_canvases[self.curr_ann_layer], self.nn_preview_canvas)

    @pyqtSlot(IntEnum)
    def analysis_menu(self, option: IntEnum) -> None:
        if option == AnalysisMenu.CIRCLES:
            if self.curr_ann_layer != 4:
                self.right_buttons.btn_ann_4.click()
            circles = find_circles(self.im_folder.curr_files[0])
            if circles is not None:
                for circle in circles[0, :]:
                    self.annotation_canvases[self.curr_ann_layer].draw_circle(circle)

        if option == AnalysisMenu.BACKGROUNDER:
            bg_folder = os.path.join(self.im_folder.folder, 'analysis/backgrounds')
            masks_folder = os.path.join(self.im_folder.folder, 'analysis/binary_masks')
            try:
                os.makedirs(bg_folder)
            except FileExistsError:
                pass
            try:
                os.makedirs(masks_folder)
            except FileExistsError:
                pass

            background_subtraction(self.im_folder.folder)
            self.im_folder._list_rois()
            self.im_folder._list_image_files()

            self.left_buttons.cb_interest.setChecked(True)
            self.left_buttons.cb_other.setChecked(False)


class ScaleBar(QLabel):
    def __init__(self, parent: QWidget, scale: float) -> None:
        super(ScaleBar, self).__init__(parent)

        # Default scale of 1px ~ 1um, adjustable here
        self.scale_factor = scale

        self.w, self.h = 190, 50
        self.setGeometry(parent.width() - self.w, parent.height() - self.h, self.w, self.h)
        self.setMinimumSize(QSize(self.w, self.h))
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self._set_canvas()
        self._draw_bars()
        self._draw_labels()

    def _set_canvas(self) -> None:
        canvas = QPixmap(self.w, self.h)
        canvas.fill(QColor('transparent'))
        # canvas.fill(QColor('red'))
        self.setPixmap(canvas)

    def _draw_bars(self) -> None:
        painter = QPainter(self.pixmap())
        painter.setPen(QPen(QColor('black'), 1, Qt.SolidLine))
        painter.setBrush(QColor('white'))
        # Mot filled
        painter.drawRect(10, 25, 10, 10)
        painter.drawRect(20, 15, 10, 10)
        painter.drawRect(30, 25, 10, 10)
        painter.drawRect(40, 15, 10, 10)
        painter.drawRect(50, 25, 40, 10)
        painter.drawRect(90, 15, 40, 10)
        painter.drawRect(130, 25, 40, 10)

        painter.setBrush(QColor('black'))
        # Filled
        painter.drawRect(10, 15, 10, 10)
        painter.drawRect(20, 25, 10, 10)
        painter.drawRect(30, 15, 10, 10)
        painter.drawRect(40, 25, 10, 10)
        painter.drawRect(50, 15, 40, 10)
        painter.drawRect(90, 25, 40, 10)
        painter.drawRect(130, 15, 40, 10)

    def _draw_labels(self) -> None:
        lab_0 = QLabel(self)
        lab_0.setText('0')
        lab_0.setAlignment(Qt.AlignCenter)
        lab_0.setGeometry(-5, 0, 30, 10)

        lab_25 = QLabel(self)
        lab_25.setText('{}'.format(40 * self.scale_factor))
        lab_25.setAlignment(Qt.AlignCenter)
        lab_25.setGeometry(35, 0, 30, 10)

        lab_50 = QLabel(self)
        lab_50.setText('{}'.format(80 * self.scale_factor))
        lab_50.setAlignment(Qt.AlignCenter)
        lab_50.setGeometry(75, 0, 30, 10)

        lab_100 = QLabel(self)
        lab_100.setText('{}'.format(160 * self.scale_factor))
        lab_100.setAlignment(Qt.AlignCenter)
        lab_100.setGeometry(155, 0, 30, 10)

        lab_um = QLabel(self)
        lab_um.setText('micrometres')
        lab_um.setAlignment(Qt.AlignCenter)
        lab_um.setGeometry(45, 40, 90, 10)
