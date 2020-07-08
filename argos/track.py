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
from argos.sortracker import SORTWidget
from argos.segwidget import SegWidget
from argos.csrtracker import CSRTWidget


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

        self._sort_widget = SORTWidget()
        self._csrt_widget = CSRTWidget()

        self._yolact_dock = qw.QDockWidget('Yolact settings')
        self._yolact_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                         qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._yolact_dock)
        self._yolact_dock.setWidget(self._yolact_widget)

        self._seg_dock = qw.QDockWidget('Segmentation settings')
        self._seg_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                         qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._seg_dock)
        self._seg_scroll = qw.QScrollArea()
        self._seg_scroll.setWidgetResizable(True)
        self._seg_scroll.setWidget(self._seg_widget)
        self._seg_dock.setWidget(self._seg_scroll)

        self._sort_dock = qw.QDockWidget('SORTracker settings')
        self._sort_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                        qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._sort_dock)
        self._sort_scroll = qw.QScrollArea()
        self._sort_scroll.setWidgetResizable(True)
        self._sort_scroll.setWidget(self._sort_widget)
        self._sort_dock.setWidget(self._sort_scroll)

        self._csrt_dock = qw.QDockWidget('CSRTracker settings')
        self._csrt_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                        qc.Qt.RightDockWidgetArea)
        self._csrt_scroll = qw.QScrollArea()
        self._csrt_scroll.setWidgetResizable(True)
        self._csrt_scroll.setWidget(self._csrt_widget)
        self._csrt_dock.setWidget(self._csrt_scroll)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._csrt_dock)
        self._yolact_action = qw.QAction('Use Yolact segmentation')
        self._seg_action = qw.QAction('Use classical segmentation')
        self._seg_grp = qw.QActionGroup(self)
        self._seg_grp.addAction(self._yolact_action)
        self._seg_grp.addAction(self._seg_action)
        self._seg_grp.setExclusive(True)
        self._yolact_action.setCheckable(True)
        self._seg_action.setCheckable(True)
        self._yolact_action.setChecked(True)
        self._seg_dock.hide()
        self._sort_action = qw.QAction('Use SORT for tracking')
        self._sort_action.setCheckable(True)
        self._csrt_action = qw.QAction('Use CSRT for tracking')
        self._csrt_action.setCheckable(True)
        self._track_grp = qw.QActionGroup(self)
        self._track_grp.addAction(self._sort_action)
        self._track_grp.addAction(self._csrt_action)
        self._sort_action.setChecked(True)
        self._debug_action = qw.QAction('Debug')
        self._debug_action.setCheckable(
            settings.value('debug', logging.ERROR) == logging.ERROR)
        self._debug_action.triggered.connect(self.setDebug)
        self._csrt_dock.hide()
        self._menubar = self.menuBar()
        self._file_menu = self._menubar.addMenu('&File')
        self._file_menu.addAction(self._video_widget.openAction)
        self._file_menu.addAction(self._video_widget.openCamAction)
        self._seg_menu = self._menubar.addMenu('&Segmentation method')
        self._seg_menu.addActions(self._seg_grp.actions())
        self._track_menu = self._menubar.addMenu('&Tracking method')
        self._track_menu.addActions(self._track_grp.actions())
        self._zoom_menu = self._menubar.addMenu('Zoom')
        self._zoom_menu.addActions([self._video_widget.zoomInAction,
                                    self._video_widget.zoomOutAction,
                                    self._video_widget.resetArenaAction])
        self._debug_menu = self.menuBar().addMenu('Advanced')
        self._debug_menu.addAction(self._debug_action)


        ##########################
        # Connections
        self._video_widget.sigSetFrame.connect(self._yolact_widget.process)
        # self._yolact_widget.sigProcessed.connect(self._sort_widget.sigTrack)
        self._yolact_widget.sigProcessed.connect(self._sort_widget.track)
        self._yolact_widget.sigProcessed.connect(self._video_widget.sigSetSegmented)
        self._seg_widget.sigProcessed.connect(self._sort_widget.sigTrack)
        self._seg_widget.sigProcessed.connect(self._video_widget.sigSetSegmented)
        self._sort_widget.sigTracked.connect(self._video_widget.setTracked)
        self._csrt_widget.sigTracked.connect(self._video_widget.setTracked)
        self._video_widget.openAction.triggered.connect(self._sort_widget.sigReset)
        self._video_widget.sigReset.connect(self._sort_widget.sigReset)
        self._video_widget.sigReset.connect(self._csrt_widget.sigReset)
        self._seg_grp.triggered.connect(self.switchSegmentation)
        self._track_grp.triggered.connect(self.switchTracking)
        self.sigQuit.connect(self._video_widget.sigQuit)
        self.sigQuit.connect(self._yolact_widget.sigQuit)
        self.sigQuit.connect(self._seg_widget.sigQuit)
        self.sigQuit.connect(self._sort_widget.sigQuit)
        self.sigQuit.connect(self._csrt_widget.sigQuit)

    @qc.pyqtSlot(bool)
    def setDebug(self, state):
        level = logging.DEBUG if state else logging.ERROR
        settings.setValue('debug', level)
        logging.getLogger().setLevel(level)

    def cleanup(self):
        self.sigQuit.emit()
        settings.sync()
        logging.debug('Saved settings')

    @qc.pyqtSlot(qw.QAction)
    def switchSegmentation(self, action):
        """Switch segmentation widget between yolact and classical"""
        self._video_widget.pauseVideo()
        if action == self._yolact_action:
            util.reconnect(self._video_widget.sigSetFrame,
                           newhandler=self._yolact_widget.process,
                           oldhandler=self._seg_widget.sigProcess)
            self._yolact_dock.setVisible(True)
            self._seg_dock.setVisible(False)
        else:  # classical segmentation, self._seg_action
            util.reconnect(self._video_widget.sigSetFrame,
                           newhandler=self._seg_widget.sigProcess,
                           oldhandler=self._yolact_widget.process)
            self._yolact_dock.setVisible(False)
            self._seg_dock.setVisible(True)

    @qc.pyqtSlot(qw.QAction)
    def switchTracking(self, action):
        """Switch tracking between SORT and CSRT"""
        self._video_widget.pauseVideo()
        if action == self._sort_action:
            self._sort_dock.show()
            self._csrt_dock.hide()
            try:
                self._video_widget.sigSetFrame.disconnect(
                    self._csrt_widget.setFrame)
            except TypeError:
                pass
            newhandler = self._sort_widget.sigTrack
            oldhandler = self._csrt_widget.setBboxes
        else:  # csrt
            self._csrt_dock.show()
            self._sort_dock.hide()
            self._video_widget.sigSetFrame.connect(self._csrt_widget.setFrame)
            newhandler = self._csrt_widget.setBboxes
            oldhandler = self._sort_widget.sigTrack
        if self._yolact_action.isChecked():
            sig = self._yolact_widget.sigProcessed
        else:
            sig = self._seg_widget.sigProcessed
        util.reconnect(sig, newhandler, oldhandler)

if __name__ == '__main__':
    app = qw.QApplication(sys.argv)
    win = ArgosMain()
    win.setMinimumSize(800, 600)
    win.setWindowTitle('Argos - track animals in video')
    win.showMaximized()
    app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())
