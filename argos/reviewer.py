# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-07-09 1:20 PM
"""Review and correct tracks"""

import sys
from typing import List, Tuple
import logging
from collections import OrderedDict
import numpy as np
import cv2
import pandas as pd
from PyQt5 import (
    QtWidgets as qw,
    QtCore as qc,
    QtGui as qg
)

import argos.constants
from argos import utility as ut
from argos.display import Scene, Display
from argos.vreader import VideoReader


settings = ut.init()


class TrackView(Display):
    def __init__(self, *args, **kwargs):
        super(TrackView, self).__init__(*args, **kwargs)

    def setViewportRect(self, rect: qc.QRectF) -> None:
        # self.fitInView(rect)  # Incorrectly sets the visible area
        self.fitInView(rect.x(), rect.y(),
                       rect.width() - rect.x(),
                       rect.height() - rect.y(),
                       qc.Qt.KeepAspectRatio)  # this works


class TrackReader(qc.QObject):
    """Class to read the tracking data"""

    sigTracks = qc.pyqtSignal(np.ndarray, int)
    sigEnd = qc.pyqtSignal()

    def __init__(self, data_file, filetype):
        super(TrackReader, self).__init__()
        self.data_path = data_file
        if filetype == 'csv':
            self.track_data = pd.read_csv(self.data_path)
        else:
            self.track_data = pd.read_hdf(self.data_path, 'tracked')

    @qc.pyqtSlot()
    def gotoFrame(self, frame_no):
        if frame_no > self.last_frame:
            self.sigEnd.emit()
            return
        self.frame_pos = frame_no
        tracks = self.track_data[self.track_data.frame == frame_no]
        tracks = tracks[['track_id', 'x', 'y', 'w', 'h']].values
        self.sigTracks.emit(tracks, frame_no)


class ReviewWidget(qw.QWidget):
    """A widget with two panes for reviewing track mislabelings"""
    sigGotoFrame = qc.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super(ReviewWidget, self).__init__(*args, **kwargs)

        self.left_frame = None
        self.left_tracks = None
        self.right_frame = None
        self.right_tracks = None
        self.frame_no = -1

        layout = qw.QVBoxLayout()
        self.before = TrackView()
        self.before.setObjectName('Left')
        self.after = TrackView()
        self.after.setObjectName('Right')
        self.before.setHorizontalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        self.before.setVerticalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        self.after.setHorizontalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        self.after.setVerticalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        self.left_list = qw.QListWidget()
        self.right_list = qw.QListWidget()
        left_layout = qw.QHBoxLayout()
        left_layout.addWidget(self.before)
        left_layout.addWidget(self.left_list)
        right_layout = qw.QHBoxLayout()
        right_layout.addWidget(self.right_list)
        right_layout.addWidget(self.after)

        panes_layout = qw.QHBoxLayout()
        panes_layout.addLayout(left_layout)
        panes_layout.addLayout(right_layout)
        layout.addLayout(panes_layout)
        self.play_button = qw.QPushButton('Play')
        self.slider = qw.QSlider(qc.Qt.Horizontal)
        self.pos_spin = qw.QSpinBox()
        ctrl_layout = qw.QHBoxLayout()
        ctrl_layout.addWidget(self.play_button)
        ctrl_layout.addWidget(self.slider)
        ctrl_layout.addWidget(self.pos_spin)
        layout.addLayout(ctrl_layout)
        self.setLayout(layout)
        self.make_actions()

    def tieViews(self, tie):
        if tie:
            logging.debug(f'Before {self.before.viewport().size()}')
            logging.debug(f'After {self.after.viewport().size()}')
            self.before.sigViewportAreaChanged.connect(self.after.setViewportRect)
            self.after.sigViewportAreaChanged.connect(self.before.setViewportRect)
        else:
            self.before.disconnect(self.before.sigViewportAreaChanged)
            self.after.disconnect(self.after.sigViewportAreaChanged)

    def make_actions(self):
        self.tieViewsAction = qw.QAction('Scroll views together')
        self.tieViewsAction.setCheckable(True)
        self.tieViewsAction.triggered.connect(self.tieViews)

    def setupReading(self, video_path, data_path, dftype):
        self.video_reader = VideoReader(video_path)
        self.track_reader = TrackReader(data_path, dftype)
        self.sigGotoFrame.connect(self.video_reader.gotoFrame)
        self.sigGotoFrame.connect(self.track_reader.gotoFrame)


    @qc.pyqtSlot(int)
    def gotoFrame(self, frame_no):
        self.frame_no = frame_no
        self.sigGotoFrame.emit()

    qc.pyqtSlot(np.ndarray, int)
    def setFrame(self, frame: np.ndarray, pos: int) -> None:
        pass

def test_reviewwidget():
    ut.init()
    logging.getLogger().setLevel(logging.DEBUG)
    app = qw.QApplication(sys.argv)
    reviewer = ReviewWidget()
    # image = cv2.imread(
    #     'C:/Users/raysu/analysis/animal_tracking/bugtracking/training_images/'
    #     'prefix_1500.png')
    image = cv2.imread('C:/Users/raysu/Documents/src/argos/test_grid.png')
    logging.debug(f'Image shape: {image.shape}')
    reviewer.before.setFrame(image, 0)
    reviewer.after.setFrame(image, 0)
    win = qw.QMainWindow()
    toolbar = win.addToolBar('Zoom')
    toolbar = win.addToolBar('Zoom')
    zi = qw.QAction('Zoom in')
    zi.triggered.connect(reviewer.before.zoomIn)
    zo = qw.QAction('Zoom out')
    zo.triggered.connect(reviewer.after.zoomOut)
    arena = qw.QAction('Select arena')
    arena.triggered.connect(reviewer.before.scene().setArenaMode)
    arena_reset = qw.QAction('Rset arena')
    arena_reset.triggered.connect(reviewer.after.scene().resetArena)
    roi = qw.QAction('Select rectangular ROIs')
    roi.triggered.connect(reviewer.before.scene().setRoiRectMode)
    poly = qw.QAction('Select polygon ROIs')
    poly.triggered.connect(reviewer.before.scene().setRoiPolygonMode)
    toolbar.addAction(zi)
    toolbar.addAction(zo)
    toolbar.addAction(arena)
    toolbar.addAction(roi)
    toolbar.addAction(poly)
    toolbar.addAction(arena_reset)
    toolbar.addAction(reviewer.tieViewsAction)
    win.setCentralWidget(reviewer)
    win.show()
    # reviewer.before.setViewportRect(qc.QRectF(50, 50, 100, 100))
    sys.exit(app.exec_())


if __name__ == '__main__':
    test_reviewwidget()

