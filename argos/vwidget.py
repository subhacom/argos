# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-05-30 11:27 AM
import sys
import logging
import threading
import numpy as np
import os
import time
from datetime import timedelta

from PyQt5 import QtWidgets as qw, QtCore as qc, QtGui as qg

from argos import utility
from argos.frameview import FrameView
from argos.vreader import VideoReader
from argos import writer

settings = utility.init()


class VidInfo(qw.QMainWindow):
    def __init__(self):
        super(VidInfo, self).__init__()
        self.setWindowTitle('Video/Data Information')
        self.vidfile_label = qw.QLabel('Video file')
        self.vidfile = qw.QLabel('')
        self.frames_label = qw.QLabel('Number of frames')
        self.frames = qw.QLabel('')
        self.fps_label = qw.QLabel('Frames per second')
        self.fps = qw.QLabel('')
        self.outfile_label = qw.QLabel('Output files')
        self.outfile = qw.QLabel('')
        self.width_label = qw.QLabel('Frame width')
        self.frame_width = qw.QLabel('')
        self.height_label = qw.QLabel('Frame height')
        self.frame_height = qw.QLabel('')

        layout = qw.QFormLayout()
        layout.addRow(self.vidfile_label, self.vidfile)
        layout.addRow(self.frames_label, self.frames)
        layout.addRow(self.fps_label, self.fps)
        layout.addRow(self.width_label, self.frame_width)
        layout.addRow(self.height_label, self.frame_height)
        widget = qw.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


class VideoWidget(qw.QWidget):
    sigSetFrame = qc.pyqtSignal(np.ndarray, int)
    sigSetTracked = qc.pyqtSignal(dict, int)
    sigSetBboxes = qc.pyqtSignal(np.ndarray, int)
    sigSetSegmented = qc.pyqtSignal(dict, int)
    sigFrameSet = qc.pyqtSignal()
    sigGotoFrame = qc.pyqtSignal(int)
    sigQuit = qc.pyqtSignal()
    sigClose = (
        qc.pyqtSignal()
    )  # separate out closing the video reader and files from thread end
    sigThreadQuit = qc.pyqtSignal()
    sigReset = qc.pyqtSignal()
    sigSetColormap = qc.pyqtSignal(str, int)
    sigArena = qc.pyqtSignal(qg.QPolygonF)
    sigVideoFile = qc.pyqtSignal(str)
    sigStatusMsg = qc.pyqtSignal(str)
    sigOutFilename = qc.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(VideoWidget, self).__init__(*args, **kwargs)
        self.currentFrame = -1
        self.display_widget = None
        self.video_reader = None
        self.segmenter = None
        self.tracker = None
        self.slider = None
        self.video_filename = ''
        self.writer = None
        self.vid_info = VidInfo()
        # As the time to process the data is generally much longer than FPS
        # using a regular interval timer puts too much backlog and the
        # play/pause functionality does not work on timer.stop() call.
        # Therefore make the timer singleshot and start it after the processing
        # finished. Also eliminates the need for keeping a wait condition in
        # reader thread
        self.timer = qc.QTimer(self)
        self.timer.setSingleShot(True)
        self.startTime = None
        self.endTime = None
        self.openAction = qw.QAction('Open video')
        self.openCamAction = qw.QAction('Open camera')
        self.playAction = qw.QAction('Play')
        self.playAction.setCheckable(True)
        self.resetAction = qw.QAction('Reset')
        self.resetAction.setToolTip(
            'Go back to the start and reset the' ' tracker'
        )
        self.refreshAction = qw.QAction('Refresh current frame')
        self.refreshAction.triggered.connect(self.refreshFrame)
        self.showFrameNumAction = qw.QAction('Show frame #')
        self.showFrameNumAction.setCheckable(True)
        self.zoomInAction = qw.QAction('Zoom in')
        self.zoomOutAction = qw.QAction('Zoom out')
        self.arenaSelectAction = qw.QAction('Select arena')
        self.rectSelectAction = qw.QAction('Select rectangles')
        self.polygonSelectAction = qw.QAction('Select polygons')
        self.resetArenaAction = qw.QAction('Reset arena')
        self.openAction.triggered.connect(self.openVideo)
        self.openCamAction.triggered.connect(self.openCamera)
        self.playAction.triggered.connect(self.playVideo)
        self.resetAction.triggered.connect(self.resetVideo)
        self.showGrayscaleAction = qw.QAction('Show in grayscale')
        self.showGrayscaleAction.setCheckable(True)
        self.setColorAction = qw.QAction('Set color')
        self.setColorAction.triggered.connect(self.setColor)
        self.autoColorAction = qw.QAction('Automatic color')
        self.autoColorAction.setCheckable(True)
        self.colormapAction = qw.QAction('Use colormap')
        self.colormapAction.setCheckable(True)
        self.colormapAction.triggered.connect(self.setColormap)
        self.setLabelInsideAction = qw.QAction('Label inside bbox')
        self.setLabelInsideAction.setCheckable(True)
        self.fontSizeAction = qw.QAction('Set font size in points')
        self.relativeFontSizeAction = qw.QAction(
            'Set font size as % of larger side of image'
        )
        self.lineWidthAction = qw.QAction('Line width')
        self.infoAction = qw.QAction('Video information')
        self.infoAction.triggered.connect(self.vid_info.show)
        self.reader_thread = qc.QThread()
        self.sigThreadQuit.connect(self.reader_thread.quit)
        self.sigQuit.connect(self.quit)
        self.reader_thread.finished.connect(self.reader_thread.deleteLater)

    @qc.pyqtSlot()
    def openCamera(self):
        self.pauseVideo()
        cam_idx, accept = qw.QInputDialog.getInt(
            self, 'Open webcam input', 'Webcam no.', 0
        )
        if not accept:
            return
        width, accept = qw.QInputDialog.getInt(
            self, 'Frame width', 'Frame width (pixels)', -1
        )
        if not accept:
            width = -1
        height, accept = qw.QInputDialog.getInt(
            self, 'Frame height', 'Frame height (pixels)', -1
        )
        if not accept:
            height = -1
        fps, accept = qw.QInputDialog.getDouble(
            self, 'Frames per second', 'Frames per second', 30
        )
        if not accept:
            fps = 30
        self.video_filename = str(cam_idx)
        directory = settings.value('video/directory', '.')
        fourcc, accept = qw.QInputDialog.getText(
            self,
            'Enter video format',
            'FOURCC code (see http://www.fourcc.org/codecs.php)',
            text='MJPG',
        )
        if not accept:
            return
        fname, _ = qw.QFileDialog.getSaveFileName(
            self, 'Save video as', directory, filter='AVI (*.avi)'
        )
        self._initIO(fname, fourcc, width=width, height=height, fps=fps)

    @qc.pyqtSlot()
    def openVideo(self):
        self.pauseVideo()
        directory = settings.value('video/directory', '.')
        fname, filter_ = qw.QFileDialog.getOpenFileName(
            self, 'Open video', directory=directory
        )
        logging.debug(f'Opening file "{fname}"')
        if len(fname) == 0:
            return
        self.video_filename = fname
        self._initIO()

    @qc.pyqtSlot()
    def setColor(self):
        self.autoColorAction.setChecked(False)
        self.colormapAction.setChecked(False)

    @qc.pyqtSlot(bool)
    def setColormap(self, check):
        if not check:
            return
        input, accept = qw.QInputDialog.getItem(
            self,
            'Colormap for track display',
            'Colormap',
            [
                'jet',
                'viridis',
                'rainbow',
                'autumn',
                'summer',
                'winter',
                'spring',
                'cool',
                'hot',
                'None',
            ],
        )
        if input == 'None':
            self.colormapAction.setChecked(False)
            return
        if not accept:
            return
        max_colors, accept = qw.QInputDialog.getInt(
            self, 'Number of colors', 'Number of colors', 10, 1, 20
        )
        if not accept:
            return
        self.autoColorAction.setChecked(False)
        self.colormapAction.setChecked(True)
        self.sigSetColormap.emit(input, max_colors)

    def _initIO(self, outfpath=None, codec=None, width=-1, height=-1, fps=30):
        # Open input
        try:
            self.video_reader = VideoReader(
                self.video_filename, width=width, height=height, fps=fps
            )
            if (
                self.video_reader.is_webcam
                and outfpath is not None
                and codec is not None
            ):
                self.video_reader.setVideoOutFile(outfpath, codec)
            logging.debug(
                f'Opened {self.video_filename} with {self.video_reader.frame_count} frames'
            )
            settings.setValue(
                'video/directory', os.path.dirname(self.video_filename)
            )
            self.currentFrame = -1
            self.startTime = None
        except IOError as err:
            qw.QMessageBox.critical(self, 'Video open failed', str(err))
            return
        ## Set-up for saving data
        self.writer = None
        directory = os.path.dirname(self.video_filename)
        filename = writer.makepath(directory, self.video_filename)
        self.outfile, _ = qw.QFileDialog.getSaveFileName(
            self, 'Save data as', filename, 'HDF5 (*.h5 *.hdf);;Text (*.csv)'
        )
        logging.info(f'Output file "{self.outfile}"')
        if len(self.outfile.strip()) == 0:
            qw.QMessageBox.critical(
                self,
                'No output file',
                'Output file not specified. Closing video',
            )
            self.video_reader = None
            self.writer = None
            return
        if self.outfile.endswith('.csv'):
            self.writer = writer.CSVWriter(self.outfile, mode='w')
            qw.QMessageBox.information(
                self,
                'Data will be saved in',
                f'{self.writer.seg_filename} and'
                f' {self.writer.track_filename}',
            )
            self.vid_info.outfile.setText(
                f'{self.writer.seg_filename} and '
                f'{self.writer.track_filename}'
            )
        else:
            self.writer = writer.HDFWriter(self.outfile, mode='w')
            qw.QMessageBox.information(
                self, 'Data will be saved in', f'{self.outfile}'
            )
            self.vid_info.outfile.setText(f'{self.outfile}')
        self.sigOutFilename.emit(self.outfile)
        self.vid_info.vidfile.setText(self.video_filename)
        self.vid_info.frames.setText(f'{self.video_reader.frame_count}')
        self.vid_info.fps.setText(f'{self.video_reader.fps}')
        self.vid_info.frame_width.setText(f'{self.video_reader.frame_width}')
        self.vid_info.frame_height.setText(f'{self.video_reader.frame_height}')

        self.sigVideoFile.emit(self.video_filename)

        settings.setValue('data/directory', os.path.dirname(self.outfile))
        ## Move the video reader to separate thread
        self.video_reader.moveToThread(self.reader_thread)

        self.timer.timeout.connect(self.video_reader.read)
        self.sigClose.connect(self.video_reader.close)

        if not self.video_reader.is_webcam:
            self.sigGotoFrame.connect(self.video_reader.gotoFrame)

        self.video_reader.sigFrameRead.connect(self.setFrame)
        self.video_reader.sigVideoEnd.connect(self.videoEnd)
        self.video_reader.sigVideoEnd.connect(self.writer.close)
        self.sigReset.connect(self.writer.reset)
        self.sigClose.connect(self.writer.close)
        self.sigSetBboxes.connect(self.writer.appendBboxes)
        self.sigSetTracked.connect(self.writer.appendTracked)

        if self.display_widget is None:
            self.display_widget = FrameView()
            self.display_widget.frameScene.setArenaMode()
            self.showGrayscaleAction.triggered.connect(
                self.display_widget.showGrayscaleAction.trigger
            )
            self.setColorAction.triggered.connect(
                self.display_widget.setColorAction.trigger
            )
            self.autoColorAction.triggered.connect(
                self.display_widget.autoColorAction.trigger
            )
            self.sigSetColormap.connect(
                self.display_widget.frameScene.setColormap
            )
            self.setLabelInsideAction.setChecked(
                self.display_widget.setLabelInsideAction.isChecked()
            )
            self.setLabelInsideAction.triggered.connect(
                self.display_widget.setLabelInsideAction.trigger
            )
            self.fontSizeAction.triggered.connect(
                self.display_widget.fontSizeAction.trigger
            )
            self.relativeFontSizeAction.triggered.connect(
                self.display_widget.relativeFontSizeAction.trigger
            )
            self.lineWidthAction.triggered.connect(
                self.display_widget.lineWidthAction.trigger
            )
            self.sigSetFrame.connect(self.display_widget.setFrame)
            self.sigSetSegmented.connect(self.display_widget.sigSetPolygons)
            self.sigSetTracked.connect(self.display_widget.setRectangles)
            self.display_widget.sigPolygonsSet.connect(self.startTimer)
            self.display_widget.sigArena.connect(self.sigArena)
            self.sigReset.connect(self.display_widget.resetArenaAction.trigger)
            self.zoomInAction.triggered.connect(self.display_widget.zoomIn)
            self.zoomOutAction.triggered.connect(self.display_widget.zoomOut)
            self.arenaSelectAction.triggered.connect(
                self.display_widget.setArenaMode
            )
            self.rectSelectAction.triggered.connect(
                self.display_widget.setRoiRectMode
            )
            self.polygonSelectAction.triggered.connect(
                self.display_widget.setRoiPolygonMode
            )
            self.resetArenaAction.triggered.connect(
                self.display_widget.resetArenaAction.trigger
            )
            # self.sigSetTracked.connect(self.displayWidget.sigSetRectangles)
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
            reset_button = qw.QToolButton()
            reset_button.setDefaultAction(self.resetAction)
            ctrl_layout.addWidget(play_button)
            ctrl_layout.addWidget(self.slider)
            ctrl_layout.addWidget(self.spinbox)
            ctrl_layout.addWidget(reset_button)
            layout = qw.QVBoxLayout()
            layout.addWidget(self.display_widget)
            layout.addLayout(ctrl_layout)
            self.setLayout(layout)
        if self.video_reader.is_webcam:
            max_frames = 1000000  # arbitrarily set maximum number of frames
        else:
            max_frames = self.video_reader.frame_count - 1
        self.spinbox.setRange(0, max_frames)
        self.slider.setRange(0, max_frames)

        self.reader_thread.start()
        if self.video_reader.is_webcam:
            self.timer.start(10)
        else:
            self.sigGotoFrame.emit(0)
        self.sigReset.emit()

    @qc.pyqtSlot()
    def videoEnd(self):
        self.pauseVideo()
        self.endTime = time.perf_counter()
        if self.startTime is not None:
            ptime = self.endTime - self.startTime
        else:
            ptime = 0
        ptime = timedelta(seconds=ptime)
        if self.outfile.endswith('.csv'):
            data_file_str = (
                f'{self.writer.seg_filename}'
                ' and {self.writer.track_filename}'
            )
        else:
            data_file_str = f'{self.outfile}'

        qw.QMessageBox.information(
            self,
            'Finished processing',
            f'Reached the end of the video {self.video_filename} '
            f'frame # {self.video_reader.frame_count}.\n'
            f'Time: {ptime}.\n'
            f'Data saved in {data_file_str}.',
        )

    @qc.pyqtSlot(dict, int)
    def setTracked(self, bboxes: dict, pos: int) -> None:
        self.sigSetTracked.emit(bboxes, pos)

    @qc.pyqtSlot()
    def startTimer(self):
        if self.playAction.isChecked():
            logging.debug(f'Starting timer for frame')
            t = int(np.round(1000.0 / self.video_reader.fps))
            t = 10 if t < 1 else t
            self.timer.start(t)

    @qc.pyqtSlot(np.ndarray, int)
    def setFrame(self, frame: np.ndarray, pos: int) -> None:
        """Handle the incoming signal from VideoReader. The `event` is for
        synchronizing with the reader so it does not swamp us with too
        many frames
        """
        self.currentFrame = pos
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
        """This function is for playing raw video without any processing"""
        if play:
            if self.startTime is None:  # indicates video was just initialized
                print('Starting at first frame')
                self.sigGotoFrame.emit(0)
            self.startTime = time.perf_counter()
            self.playAction.setText('Pause')
            t = int(np.round(1000.0 / self.video_reader.fps))
            t = 10 if t < 1 else t
            self.timer.start(t)
        else:
            self.pauseVideo()

    @qc.pyqtSlot()
    def pauseVideo(self):
        self.playAction.setChecked(False)
        self.timer.stop()
        self.playAction.setText('Play')
        self.endTime = time.perf_counter()
        if self.startTime is not None:
            ptime = timedelta(seconds=self.endTime - self.startTime)
            msg = (
                f'Processed till frame # {self.currentFrame}'
                f' in time: {ptime}.'
            )
            logging.info(msg)
            self.sigStatusMsg.emit(msg)

    @qc.pyqtSlot()
    def resetVideo(self):
        self.pauseVideo()
        self.sigReset.emit()
        self.startTime = None
        self.sigGotoFrame.emit(0)

    @qc.pyqtSlot()
    def refreshFrame(self):
        self.sigGotoFrame.emit(self.currentFrame)
        self.slider.blockSignals(True)
        self.slider.setValue(self.currentFrame)
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(self.currentFrame)
        self.spinbox.blockSignals(False)

    @qc.pyqtSlot()
    def quit(self):
        self.sigClose.emit()
        self.sigThreadQuit.emit()
        self.reader_thread.wait()


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
