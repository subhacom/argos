# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-05-30 11:27 AM
import sys
import logging
import threading
import numpy as np

from PyQt5 import QtWidgets as qw, QtCore as qc

from argos import utility
from argos.display import Display
from argos.vreader import VideoReader


class VideoWidget(qw.QWidget):

    sigSetFrame = qc.pyqtSignal(np.ndarray, int)
    sigSetBboxes = qc.pyqtSignal(dict, int)
    sigFrameSet = qc.pyqtSignal()
    sigGotoFrame = qc.pyqtSignal(int)
    sigQuit = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(VideoWidget, self).__init__(*args, **kwargs)
        self.display_widget = None
        self.video_reader = None
        self.segmenter = None
        self.tracker = None
        self.slider = None
        self.video_filename = ''
        # As the time to process the data is generally much longer than FPS
        # using a regular interval timer puts too much backlog and the
        # play/pause functionality does not work on timer.stop() call.
        # Therefore make the timer singleshot and start it after the processing
        # finished. Also eliminates the need for keeping a wait condition in
        # reader thread
        self.timer = qc.QTimer(self)
        self.timer.setSingleShot(True)
        self.openAction = qw.QAction('Open video')
        self.playAction = qw.QAction('Play/Pause')
        self.playAction.setCheckable(True)
        self.openAction.triggered.connect(self.openVideo)
        self.playAction.triggered.connect(self.playVideo)
        self.reader_thread = qc.QThread()
        self.sigQuit.connect(self.reader_thread.quit)
        self.reader_thread.finished.connect(self.reader_thread.deleteLater)

    @qc.pyqtSlot()
    def openVideo(self):
        self.timer.stop()
        fname = qw.QFileDialog.getOpenFileName(self, 'Open video')
        logging.debug(f'Opening file "{fname}"')
        if len(fname[0]) == 0:
            return
        try:
            self.video_reader = VideoReader(fname[0])
            logging.debug(f'Opened {fname[0]} with {self.video_reader.frame_count} frames')
        except IOError as err:
            qw.QMessageBox.critical('Video open failed', str(err))
        self.video_filename = fname[0]

        self.video_reader.moveToThread(self.reader_thread)

        self.timer.timeout.connect(self.video_reader.read)
        self.sigGotoFrame.connect(self.video_reader.gotoFrame)
        self.video_reader.sigFrameRead.connect(self.setFrame)
        self.video_reader.sigVideoEnd.connect(self.pauseVideo)
        if self.display_widget is None:
            self.display_widget = Display()
            self.sigSetFrame.connect(self.display_widget.setFrame)
            self.sigSetBboxes.connect(
                self.display_widget.setRectangles)
            # self.sigSetBboxes.connect(self.display_widget.sigSetRectangles)
            self.slider = qw.QSlider(qc.Qt.Horizontal)
            self.slider.valueChanged.connect(self.sigGotoFrame)
            self.spinbox = qw.QSpinBox()
            self.spinbox.valueChanged.connect(self.sigGotoFrame)
            ctrl_layout = qw.QHBoxLayout()
            open_button = qw.QToolButton()
            open_button.setDefaultAction(self.openAction)
            ctrl_layout.addWidget(open_button)
            play_button = qw.QToolButton()
            play_button.setDefaultAction(self.playAction)
            ctrl_layout.addWidget(play_button)
            ctrl_layout.addWidget(self.slider)
            ctrl_layout.addWidget(self.spinbox)
            layout = qw.QVBoxLayout()
            layout.addWidget(self.display_widget)
            layout.addLayout(ctrl_layout)
            self.setLayout(layout)
        self.slider.setRange(0, self.video_reader.frame_count-1)
        self.spinbox.setRange(0, self.video_reader.frame_count-1)
        self.reader_thread.start()
        self.sigGotoFrame.emit(0)

    # TODO - for testing yolact part - intermediate trial before adding tracker
    @qc.pyqtSlot(dict, int)
    def setBboxes(self, bboxes: dict, pos: int) -> None:
        logging.debug(f'Here')
        self.sigSetBboxes.emit(bboxes, pos)
        logging.debug(f'Set wait condition for frame {pos}')
        if self.playAction.isChecked():
            self.timer.start(1000.0 / self.video_reader.fps)

    @qc.pyqtSlot(np.ndarray, int)
    def setFrame(self, frame: np.ndarray, pos: int) -> None:
        """Handle the incoming signal from VideoReader. The `event` is for
        synchronizing with the reader so it does not swamp us with too
        many frames

        TODO move this to a helper class to send frame to Tracker and
        synchronize tracked bboxes to be overlayed with frame on Display
        """
        self.sigSetFrame.emit(frame, pos)
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(pos)
        self.spinbox.blockSignals(False)
        self.sigFrameSet.emit()

    @qc.pyqtSlot(bool)
    def playVideo(self, play: bool) -> None:
        """This function is for playing raw video without any processing
        """
        if play:
            time = 1000.0 / self.video_reader.fps
            self.timer.start(time)

    @qc.pyqtSlot()
    def pauseVideo(self):
        self.playAction.setChecked(False)
        self.timer.stop()


def test_vreader():
    utility.init()
    app = qw.QApplication([])
    widget = VideoWidget()
    win = qw.QMainWindow()
    toolbar = win.addToolBar('Play')
    toolbar.addAction(widget.openAction)
    toolbar.addAction(widget.playAction)
    win.setCentralWidget(widget)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    test_vreader()