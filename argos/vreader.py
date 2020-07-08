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
    sigVideoEnd = qc.pyqtSignal()

    def __init__(self, path, waitCond=None):
        super(VideoReader, self).__init__()
        # TODO check if I really need the mutex just for reading
        self.mutex = qc.QMutex()
        locker = qc.QMutexLocker(self.mutex)   # this ensures unlock at exit
        self.is_webcam = path.isdigit()
        self._path = path
        self._outpath = None
        self._outfile = None
        self._ts_file = None
        self._waitCond = waitCond
        if self.is_webcam:
            # Camera FPS opens a temporary VideoCapture with the camera
            # - so do that before initializing current VideoCapture
            self.fps = cu.get_camera_fps(int(path))
            self._vid = cv2.VideoCapture(int(path))
            self.frame_count = -1
            ret, frame = self._vid.read()
            logging.debug(f'Read frame: {frame.shape}')
            self.frame_height, self.frame_width = frame.shape[:2]
            self._outpath = f'{path}.avi'
        else:
            self._vid = cv2.VideoCapture(path)
            self.frame_count = int(self._vid.get(cv2.CAP_PROP_FRAME_COUNT))
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
        self._ts_file = open(f'{self._outpath}.csv', 'w')
        self._ts_writer = csv.writer(self._ts_file)
        self._ts_writer.writerow(['frame', 'timestamp'])
        self._frame_no = -1
        self.mutex.unlock()

    @qc.pyqtSlot(int)
    def gotoFrame(self, frame_no: int) -> None:
        logging.debug(f'goto frame {frame_no}, called from {self.sender()}')
        # QMutexLocker is a convenience class to keep mutex locked until the
        # locker is deleted
        self.mutex.lock()
        self._vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self.mutex.unlock()
        self.read()

    @qc.pyqtSlot()
    def read(self):
        """Read a single frame"""
        logging.debug(f'Starting read, triggered by {self.sender()}')
        self.mutex.lock()
        pos = int(self._vid.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self._vid.read()
        if self._waitCond is not None:
            self._waitCond.clear()
        self.mutex.unlock()
        if not ret:
            logging.debug('Video at end')
            self.sigVideoEnd.emit()
            return

        if frame is not None:
            self.mutex.lock()
            # event = threading.Event()
            if self.is_webcam and self._outfile is not None:
                self._frame_no += 1
                pos = self._frame_no
                self._outfile.write(frame)
                ts = datetime.now()
                self._ts_writer.writerow([self._frame_no, ts])
                logging.debug(f'Wrote timestamp for frame {self._frame_no}: {ts}')
            self.sigFrameRead.emit(frame.copy(), pos)
            logging.debug(f'Read frame {pos}')
            if self._waitCond is not None:
                self._waitCond.wait()
            self.mutex.unlock()

            # event.wait()
            logging.debug('Finished waiting')


    def __del__(self):
        if self._vid.isOpened():
            self._vid.release()
        if self._outfile is not None and self._outfile.isOpened():
            self._outfile.release()
            self._ts_file.close()
        # logging.debug('Destructor of video reader')

