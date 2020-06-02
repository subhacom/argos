# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-05-28 3:01 PM

import sys
import time
import enum
import threading
import collections
import logging

import argos.utility
import numpy as np
import cv2
from PyQt5 import (
    QtWidgets as qw,
    QtCore as qc,
    QtGui as qg
)


import argos.utility as util
from argos.vwidget import VideoWidget
from argos.yolactwidget import YolactWidget

# from argos.drawingtools import DrawingTools, ArenaFilter
# from argos.conversions import cv2qimage
# from argos import (
#     segwidgets,
#     frameprocessor,
#     batchprocessor,
#     yolactwidget,
#     tracking,
#     trackwidgets,
#     utility)

# Set up logging for multithreading/multiprocessing
settings = util.init()


class ArgosMain(qw.QMainWindow):
    """"Main user interface"""

    def __init__(self, *args, **kwargs):
        super(ArgosMain, self).__init__(*args, **kwargs)
        self._video_widget = VideoWidget()
        self.setCentralWidget(self._video_widget)
        self._yolact_widget = YolactWidget()
        self._video_widget.setSegmenter(self._yolact_widget)
        self._yolact_dock = qw.QDockWidget('Yolact settings')
        self._yolact_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                         qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._yolact_dock)
        self._yolact_dock.setWidget(self._yolact_widget)

        self._menubar = self.menuBar()
        self._file_menu = self._menubar.addMenu('&File')
        self._file_menu.addAction(self._video_widget.openAction)

    def cleanup(self):
        # self.video_reader.terminate()
        # self.frame_processor.terminate()
        settings.sync()
        logging.debug('Saved settings')


if __name__ == '__main__':
    app = qw.QApplication(sys.argv)
    win = ArgosMain()
    win.setMinimumSize(800, 600)
    win.setWindowTitle('Argos - track animals in video')
    win.showMaximized()
    app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())
