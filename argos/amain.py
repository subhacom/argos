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
from argos.tracker import SORTWidget
from argos.segwidget import SegWidget

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

    sigQuit = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(ArgosMain, self).__init__(*args, **kwargs)
        self._video_widget = VideoWidget()
        self.setCentralWidget(self._video_widget)
        self._yolact_widget = YolactWidget()
        self._seg_widget = SegWidget()

        self._tracker_widget = SORTWidget()

        self._yolact_dock = qw.QDockWidget('Yolact settings')
        self._yolact_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                         qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._yolact_dock)
        self._yolact_dock.setWidget(self._yolact_widget)

        self._seg_dock = qw.QDockWidget('Segmentation settings')
        self._seg_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                         qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._seg_dock)
        self._seg_dock.setWidget(self._seg_widget)

        self._tracker_dock = qw.QDockWidget('Tracker settings')
        self._tracker_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                          qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._tracker_dock)
        self._tracker_dock.setWidget(self._tracker_widget)

        self._yolact_action = qw.QAction('Use Yolact segmentation')
        self._seg_action = qw.QAction('Use classical segmentation')
        self._seg_grp = qw.QActionGroup(self)
        self._seg_grp.addAction(self._yolact_action)
        self._seg_grp.addAction(self._seg_action)
        self._seg_grp.setExclusive(True)
        self._yolact_action.setCheckable(True)
        self._seg_action.setCheckable(True)
        self._yolact_action.setChecked(True)
        self._seg_dock.setVisible(False)

        self._menubar = self.menuBar()
        self._file_menu = self._menubar.addMenu('&File')
        self._file_menu.addAction(self._video_widget.openAction)
        self._seg_menu = self._menubar.addMenu('&Segmentation method')
        self._seg_menu.addActions(self._seg_grp.actions())


        ##########################
        # Connections
        self._video_widget.sigSetFrame.connect(self._yolact_widget.process)
        self._yolact_widget.sigProcessed.connect(self._tracker_widget.sigTrack)
        self._yolact_widget.sigProcessed.connect(self._video_widget.sigSetSegmented)
        self._seg_widget.sigProcessed.connect(self._tracker_widget.sigTrack)
        self._seg_widget.sigProcessed.connect(self._video_widget.sigSetSegmented)
        self._tracker_widget.sigTracked.connect(self._video_widget.setTracked)
        self._video_widget.openAction.triggered.connect(self._tracker_widget.sigReset)
        self._video_widget.sigReset.connect(self._tracker_widget.sigReset)
        self._seg_grp.triggered.connect(self.switchSegmentation)
        self.sigQuit.connect(self._video_widget.sigQuit)
        self.sigQuit.connect(self._yolact_widget.sigQuit)
        self.sigQuit.connect(self._seg_widget.sigQuit)
        self.sigQuit.connect(self._tracker_widget.sigQuit)

    def cleanup(self):
        settings.sync()
        logging.debug('Saved settings')
        self.sigQuit.emit()

    @qc.pyqtSlot(qw.QAction)
    def switchSegmentation(self, action):
        """Switch segmentation widget between yolact and classical"""
        self._video_widget.sigSetFrame.disconnect()
        if action == self._yolact_action:
            self._video_widget.sigSetFrame.connect(self._yolact_widget.process)
            self._yolact_dock.setVisible(True)
            self._seg_dock.setVisible(False)
        else:  # classical segmentation, self._seg_action
            self._video_widget.sigSetFrame.connect(self._seg_widget.sigProcess)
            self._yolact_dock.setVisible(False)
            self._seg_dock.setVisible(True)



if __name__ == '__main__':
    app = qw.QApplication(sys.argv)
    win = ArgosMain()
    win.setMinimumSize(800, 600)
    win.setWindowTitle('Argos - track animals in video')
    win.showMaximized()
    app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())
