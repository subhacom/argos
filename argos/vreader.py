# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-05-28 3:03 PM

import sys
import logging
import threading
import numpy as np
import cv2
from PyQt5 import QtCore as qc


class VideoReader(qc.QThread):
    """Utility to read the video frames.

    Needs a separate thread to avoid blocking the main UI"""

    sigFrameRead = qc.pyqtSignal(np.ndarray, int)
    sigVideoEnd = qc.pyqtSignal()

    def __init__(self, path, waitCond=None):
        super(VideoReader, self).__init__()
        # TODO check if I really need the mutex just for reading
        self.mutex = qc.QMutex()
        locker = qc.QMutexLocker(self.mutex)   # this ensures unlock at exit
        self._vid = cv2.VideoCapture(path)
        self._waitCond = waitCond
        if not self._vid.isOpened():
            raise IOError(f'Could not open video: {path}')
        self._path = path
        self.is_webcam = path.isdigit()
        if self.is_webcam:
            self.frame_count = -1
            self.fps = -1
        else:
            self.frame_count = int(self._vid.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self._vid.get(cv2.CAP_PROP_FPS)

    def setWaitCond(self, waitCond: threading.Event) -> None:
        self._waitCond = waitCond

    @qc.pyqtSlot(int)
    def gotoFrame(self, frame_no: int) -> None:
        logging.debug(f'goto frame {frame_no}, called from {self.sender()}')
        # QMutexLocker is a convenience class to keep mutex locked until the
        # locker is deleted
        self.mutex.lock()
        self._vid.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        self.mutex.unlock()

    def run(self):
        """Read a single frame"""
        logging.debug(f'Starting read')
        self.mutex.lock()
        pos = int(self._vid.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self._vid.read()
        if self._waitCond is not None:
            self._waitCond.clear()
        self.mutex.unlock()
        if not ret:
            logging.debug('Video at end')
            self.sigVideoEnd.emit()
            self.sigFinished.emit()
            return

        if frame is not None:
            self.mutex.lock()
            # event = threading.Event()
            self.sigFrameRead.emit(frame.copy(), pos)
            logging.debug(f'Read frame {pos}')
            if self._waitCond is not None:
                self._waitCond.wait()
            self.mutex.unlock()

            # event.wait()
            logging.debug('Finished waiting')

    def quit(self):
        self._vid.release()
        super(VideoReader, self).quit()

    def __del__(self):
        self._vid.release()
        self.wait()
        # logging.debug('Destructor of video reader')
