# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-05-28 3:03 PM

import sys
import logging
import threading
import numpy as np
import csv
from datetime import datetime
import cv2
from PyQt5 import QtCore as qc

from argos import utility as ut
from argos import capture as cu


class VideoReader(qc.QObject):
    """Utility to read the video frames.

    Needs a separate thread to avoid blocking the main UI"""

    sigFrameRead = qc.pyqtSignal(np.ndarray, int)
    sigSeekError = qc.pyqtSignal(Exception)
    sigVideoEnd = qc.pyqtSignal()

    def __init__(self, path: str, width=-1, height=-1, fps=30, waitCond: threading.Event=None):
        super(VideoReader, self).__init__()
        # TODO check if I really need the mutex just for reading
        self.mutex = qc.QMutex()
        self.is_webcam = path.isdigit()
        self._path = path
        self._outpath = None
        self._outfile = None
        self._ts_file = None
        self._waitCond = waitCond
        self._frame_no = -1
        self.frame_width = width
        self.frame_height = height
        if self.is_webcam:
            # Camera FPS opens a temporary VideoCapture with the camera
            # - so do that before initializing current VideoCapture
            self.mutex.lock()            
            self.fps, self.frame_width, self.frame_height = cu.get_camera_fps(int(path), width, height, fps, nframes=30)
            self.mutex.unlock()
            print(f'Camera settings: w={self.frame_width}, h={self.frame_height}, fps={self.fps}')
            self._vid = cv2.VideoCapture(int(path))
            self._vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self._vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self._vid.set(cv2.CAP_PROP_FPS, fps)
            self.frame_count = -1
            self._outpath = f'{path}.avi'
        else:
            self._vid = cv2.VideoCapture(path)
            self.frame_count = int(self._vid.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_height = int(self._vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_width = int(self._vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.fps = self._vid.get(cv2.CAP_PROP_FPS)
        if not self._vid.isOpened():
            raise IOError(f'Could not open video: {path}')


    @qc.pyqtSlot(str)
    def setVideoOutFile(self, path, format):
        self.mutex.lock()
        self._outpath = path
        self._fourcc = cv2.VideoWriter_fourcc(*format)
        self._outfile = cv2.VideoWriter(self._outpath, self._fourcc, self.fps,
                                      (self.frame_width, self.frame_height))
        try:
            self._ts_file = open(f'{self._outpath}.csv', 'w')
            self._ts_writer = csv.writer(self._ts_file)
            self._ts_writer.writerow(['frame', 'timestamp'])
            self._frame_no = -1
        finally:
            self.mutex.unlock()

    @qc.pyqtSlot(int)
    def gotoFrame(self, frame_no: int) -> None:
        logging.debug(f'goto frame {frame_no}, called from {self.sender()}')
        # QMutexLocker is a convenience class to keep mutex locked until the
        # locker is deleted
        self.mutex.lock()
        # OpenCV uses ffmpeg which results in a nasty bug: https://github.com/opencv/opencv/issues/9053
        # In short, jumping to a specific frame no is completely unreliable and depends on the video format and codec
        if self._waitCond is not None:
            self._waitCond.clear()
        if not self.is_webcam:
            if frame_no >= self._vid.get(cv2.CAP_PROP_FRAME_COUNT):
                logging.info(f'Video at end @ {datetime.now()}')
                self.mutex.unlock()
                self.sigVideoEnd.emit()
                return
            self._vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            pos = int(self._vid.get(cv2.CAP_PROP_POS_FRAMES))
            if pos != frame_no:
                self.mutex.unlock()
                self.sigSeekError.emit(RuntimeError(
                    f'This video format does not allow correct seek: '
                    f'tried {frame_no}, got {pos}'))
                return
        self._frame_no = frame_no
        ret, frame = self._vid.read()
        self.mutex.unlock()
        if not ret:
            logging.debug(f'Video at end @{datetime.now()}')
            self.sigVideoEnd.emit()
            return
        if frame is None:
            logging.info(f'Empty frame at position {frame_no} @ {datetime.now()}')
            self.sigVideoEnd.emit()
            return
        if self.is_webcam and self._outfile is not None:
            # self._frame_no += 1
            # pos = self._frame_no
            self._outfile.write(frame)
            ts = datetime.now().isoformat()
            self._ts_writer.writerow([self._frame_no, ts])
            # logging.debug(f'Wrote timestamp for frame {self._frame_no}: {ts}')
        if self._frame_no == 0:
            logging.info(f'First frame read @ {datetime.now()}')
        self.sigFrameRead.emit(frame.copy(), pos)
        if self._waitCond is not None:
            self._waitCond.wait()
        # event.wait()
        logging.debug(f'Finished waiting on frame {pos}')

    @qc.pyqtSlot()
    def read(self):
        """Read a single frame"""
        logging.debug(f'Starting read, triggered by {self.sender()}')
        self.mutex.lock()
        ret, frame = self._vid.read()
        if self._waitCond is not None:
            self._waitCond.clear()
        self.mutex.unlock()
        if not ret:
            logging.info(f'Video at end @ {datetime.now()}')
            self.sigVideoEnd.emit()
            return
        if frame is None:
            logging.info(f'Empty frame at position {self._frame_no}'
                         f' @ {datetime.now()}')
            self.sigVideoEnd.emit()
            return
        # event = threading.Event()
        self._frame_no += 1
        if self._frame_no == 0:
            logging.info(f'Read first frame @ {datetime.now()}')
        if self.is_webcam and self._outfile is not None:
            assert self.frame_width == frame.shape[1] and self.frame_height == frame.shape[0]
            self._outfile.write(frame)
            ts = datetime.now()
            self._ts_writer.writerow([self._frame_no, ts])
            # logging.debug(f'Wrote timestamp for frame {self._frame_no}: {ts}')
        self.sigFrameRead.emit(frame.copy(), self._frame_no)
        # logging.debug(f'Read frame {self._frame_no}')
        if self._waitCond is not None:
            self._waitCond.wait()
        # event.wait()
        logging.debug('Finished waiting')

    @qc.pyqtSlot()
    def close(self):
        print('VideoReader: Quitting')
        if self._vid.isOpened():
            self._vid.release()
        if self._outfile is not None and self._outfile.isOpened():
            self._outfile.release()
            self._ts_file.close()
        if self._vid.isOpened():
            self._vid.release()
        if self._outfile is not None and self._outfile.isOpened():
            self._outfile.release()
            self._ts_file.close()
