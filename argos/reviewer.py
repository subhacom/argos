# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-07-09 1:20 PM
"""Review and correct tracks"""

import sys
from typing import List, Tuple
import logging
from collections import defaultdict
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
    sigSelected = qc.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(TrackView, self).__init__(*args, **kwargs)
        self.sigSelected.connect(self.scene().setSelected)

    def setViewportRect(self, rect: qc.QRectF) -> None:
        self.fitInView(rect.x(), rect.y(),
                       rect.width(),
                       rect.height(),
                       qc.Qt.KeepAspectRatio)

    @qc.pyqtSlot(str)
    def setSelected(self, id_: str) -> None:
        self.sigSelected.emit([int(id_)])


class TrackReader(qc.QObject):
    """Class to read the tracking data"""
    sigEnd = qc.pyqtSignal()

    def __init__(self, data_file, filetype):
        super(TrackReader, self).__init__()
        self.data_path = data_file
        if filetype == 'csv':
            self.track_data = pd.read_csv(self.data_path)
        else:
            self.track_data = pd.read_hdf(self.data_path, 'tracked')
        self.last_frame = int(self.track_data.frame.max())
        self._undo_dict = defaultdict(dict)
        self.to_delete = set()

    def getTracks(self, frame_no):
        if frame_no > self.last_frame:
            self.sigEnd.emit()
            return
        self.frame_pos = frame_no
        tracks = self.track_data[self.track_data.frame == frame_no]
        tracks = {int(row.trackid): row[['x', 'y', 'w', 'h']].values
                  for index, row in tracks.iterrows()
                  if int(row.trackid) not in self.to_delete}
        return tracks

    def changeTrack(self, frame_no, orig_id, new_id):
        """When user assigns `new_id` to `orig_id` keep it in undo buffer"""
        self._undo_dict[frame_no][orig_id] = new_id

    def undoChangeTrack(self, frame_no):
        self._undo_dict.pop(frame_no, None)

    def consolidateChanges(self, filepath):
        """Consolidate all the changes made in track id assignment.

        Assumptions: as tracking progresses, only new, bigger numbers are
        assigned for track ids. track_id never goes down.

        track ids can be swapped.
        """
        assignments = {}
        frame_list = sorted(self._undo_dict.keys())
        if len(frame_list) == 0:
            return
        assignments = self._undo_dict[frame_list[0]]
        for frame_no in frame_list[1:]:
            next = self._undo_dict[frame_no]
            for key, value in next.items():
                if key in assignments:
                    raise Exception('Key already present in assignments ...')
                while value in assignments:
                    value = assignments[value]
                assignments[key] = value
                next[key] = value
        data = []
        for frame_no, tdata in self.track_data.groupby('frame'):
            mapping = self._undo_dict.pop(frame_no, {})
            for index, row in tdata.iterrows():
                trackid = int(row.trackid)
                if trackid in self.to_delete:
                    continue
                data.append({'frame': frame_no,
                             'trackid': mapping.get(trackid, trackid),
                             'x': row.x,
                             'y': row.y,
                             'w': row.w,
                             'h': row.h
                             })
        data = pd.DataFrame(data=data)
        data.to_csv(filepath)
        self.track_data = data
        self._undo_dict = defaultdict(dict)

    def markForDeletion(self, track_id):
        self.to_delete.add(track_id)


class ReviewWidget(qw.QWidget):
    """A widget with two panes for reviewing track mislabelings"""
    sigGotoFrame = qc.pyqtSignal(int)
    sigLeftFrame = qc.pyqtSignal(np.ndarray, int)
    sigRightFrame = qc.pyqtSignal(np.ndarray, int)
    sigLeftTracks = qc.pyqtSignal(dict)
    sigRightTracks = qc.pyqtSignal(dict)

    def __init__(self, *args, **kwargs):
        super(ReviewWidget, self).__init__(*args, **kwargs)

        self.left_frame = None
        self.left_tracks = None
        self.right_frame = None
        self.right_tracks = None
        self.frame_no = -1
        self.timer = qc.QTimer(self)
        self.timer.setSingleShot(True)
        self.video_reader = None
        self.track_reader = None
        self.left_tracks = {}
        self.right_tracks = {}

        layout = qw.QVBoxLayout()
        self.left_view = TrackView()
        self.left_view.setObjectName('Left')
        self.right_view = TrackView()
        self.right_view.setObjectName('Right')
        self.left_view.setHorizontalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        self.left_view.setVerticalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        self.right_view.setHorizontalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        self.right_view.setVerticalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)

        self.left_list = qw.QListWidget()
        self.right_list = qw.QListWidget()

        left_layout = qw.QHBoxLayout()
        left_layout.addWidget(self.left_view)
        left_layout.addWidget(self.left_list)
        right_layout = qw.QHBoxLayout()
        right_layout.addWidget(self.right_list)
        right_layout.addWidget(self.right_view)

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
        self.makeShortcuts()

        self.timer.timeout.connect(self.nextFrame)
        self.sigLeftFrame.connect(self.left_view.setFrame)
        self.sigRightFrame.connect(self.right_view.setFrame)
        self.sigLeftTracks.connect(self.left_view.sigSetRectangles)
        self.sigRightTracks.connect(self.right_view.sigSetRectangles)
        self.left_list.currentTextChanged.connect(self.left_view.setSelected)
        self.right_list.currentTextChanged.connect(self.right_view.setSelected)
        self.play_button.clicked.connect(self.playVideo)
        self.play_button.setCheckable(True)

    def tieViews(self, tie):
        if tie:
            logging.debug(f'Before {self.left_view.viewport().size()}')
            logging.debug(f'After {self.right_view.viewport().size()}')
            self.left_view.sigViewportAreaChanged.connect(
                self.right_view.setViewportRect)
            self.right_view.sigViewportAreaChanged.connect(
                self.left_view.setViewportRect)
        else:
            self.left_view.disconnect(self.left_view.sigViewportAreaChanged)
            self.right_view.disconnect(self.right_view.sigViewportAreaChanged)

    def make_actions(self):
        self.tieViewsAction = qw.QAction('Scroll views together')
        self.tieViewsAction.setCheckable(True)
        self.tieViewsAction.triggered.connect(self.tieViews)
        self.autoColorAction = qw.QAction('Automatic color')
        self.autoColorAction.setCheckable(True)
        self.autoColorAction.triggered.connect(
            self.left_view.autoColorAction.trigger)
        self.autoColorAction.triggered.connect(
            self.right_view.autoColorAction.trigger)
        self.openAction = qw.QAction('Open tracked data')
        self.openAction.triggered.connect(self.openTrackedData)

    def makeShortcuts(self):
        self.sc_zoom_in = qw.QShortcut(qg.QKeySequence('+'), self)
        self.sc_zoom_in.activated.connect(self.left_view.zoomIn)
        self.sc_zoom_in.activated.connect(self.right_view.zoomIn)
        self.sc_zoom_out = qw.QShortcut(qg.QKeySequence('-'), self)
        self.sc_zoom_out.activated.connect(self.left_view.zoomOut)
        self.sc_zoom_out.activated.connect(self.right_view.zoomOut)

        self.sc_next = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_PageDown), self)
        self.sc_next.activated.connect(self.nextFrame)
        self.sc_prev = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_PageUp), self)
        self.sc_prev.activated.connect(self.prevFrame)

        self.sc_remove = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Delete), self)
        self.sc_remove.activated.connect(
            self.deleteSelected)
        self.sc_remove_2 = qw.QShortcut(qg.QKeySequence('X'), self)
        self.sc_remove_2.activated.connect(
            self.deleteSelected)

    def deleteSelected(self):
        if qw.QApplication.focusWidget() == self.left_list:
            items = self.left_list.selectedItems()
        elif qw.QApplication.focusWidget() == self.right_list:
            items = self.right_list.selectedItems()
        selected = [int(item.text()) for item in items]
        self.left_view.scene().setSelected(selected)
        self.right_view.scene().setSelected(selected)
        self.left_view.scene().removeSelected()
        self.right_view.scene().removeSelected()
        for sel in selected:
            self.track_reader.markForDeletion(sel)
            left_items = self.left_list.findItems(items[0].text(),
                                                  qc.Qt.MatchExactly)
            for item in left_items:
                self.left_list.takeItem(self.left_list.row(item))
            right_items = self.right_list.findItems(items[0].text(),
                                                  qc.Qt.MatchExactly)
            for item in right_items:
                self.right_list.takeItem(self.right_list.row(item))

    def setupReading(self, video_path, data_path, dftype):
        self.video_reader = VideoReader(video_path)
        self.track_reader = TrackReader(data_path, dftype)
        self.sigGotoFrame.connect(self.video_reader.gotoFrame)
        self.video_reader.sigFrameRead.connect(self.setFrame)
        self.frame_interval = 1000.0 / self.video_reader.fps

    @qc.pyqtSlot(int)
    def gotoFrame(self, frame_no):
        if frame_no >= self.track_reader.last_frame or \
                self.video_reader is None:
            return
        self.frame_no = frame_no
        self.sigGotoFrame.emit(self.frame_no)
        self.sigGotoFrame.emit(self.frame_no + 1)

    @qc.pyqtSlot()
    def nextFrame(self):
        self.gotoFrame(self.frame_no + 1)
        if self.play_button.isChecked():
            self.timer.start(self.frame_interval)

    def prevFrame(self):
        if self.frame_no >= 1:
            self.gotoFrame(self.frame_no - 1)

    @qc.pyqtSlot(np.ndarray, int)
    def setFrame(self, frame: np.ndarray, pos: int) -> None:
        if pos == self.frame_no:
            self.sigLeftFrame.emit(frame, pos)
            self.left_tracks = self.track_reader.getTracks(pos)
            self.sigLeftTracks.emit(self.left_tracks)
            self.left_list.clear()
            self.left_list.addItems([str(x) for x in sorted(self.left_tracks.keys())])
        elif pos == self.frame_no + 1:
            self.sigRightFrame.emit(frame, pos)
            self.right_tracks = self.track_reader.getTracks(pos)
            self.sigRightTracks.emit(self.right_tracks)
            self.right_list.clear()
            self.right_list.addItems([str(x) for x in sorted(self.right_tracks.keys())])
        else:
            raise Exception('This should not be reached')
        left_keys = set(self.left_tracks.keys())
        right_keys = set(self.right_tracks.keys())
        if left_keys != right_keys:
            self.play_button.setChecked(False)
            self.playVideo(False)
            logging.info(f'Tracks don\'t match in frame {pos}: '
                         f'{left_keys.symmetric_difference(right_keys)}')

    @qc.pyqtSlot()
    def openTrackedData(self):
        datadir = settings.value('data/directory', '.')
        track_filename, filter = qw.QFileDialog.getOpenFileName(
            self,
            'Open tracked data',
            datadir, filter='HDF5 (*.h5 *.hdf);; Text (*.csv)')
        logging.debug(f'filename:{track_filename}\nselected filter:{filter}')
        viddir = settings.value('video/directory', '.')
        vid_filename, vfilter = qw.QFileDialog.getOpenFileName(
            self, 'Open video', viddir)
        logging.debug(f'filename:{vid_filename}\nselected filter:{vfilter}')
        self.setupReading(vid_filename, track_filename, filter)

    @qc.pyqtSlot(bool)
    def playVideo(self, play):
        if play:
            self.play_button.setText('Pause')
            self.timer.start(self.frame_interval)
        else:
            self.play_button.setText('Play')
            self.timer.stop()


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
    reviewer.left_view.setFrame(image, 0)
    reviewer.right_view.setFrame(image, 0)
    win = qw.QMainWindow()
    win.setCentralWidget(reviewer)
    win.show()
    # reviewer.before.setViewportRect(qc.QRectF(50, 50, 100, 100))
    sys.exit(app.exec_())


def test_review():
    ut.init()
    logging.getLogger().setLevel(logging.DEBUG)
    app = qw.QApplication(sys.argv)
    reviewer = ReviewWidget()
    # image = cv2.imread(
    #     'C:/Users/raysu/analysis/animal_tracking/bugtracking/training_images/'
    #     'prefix_1500.png')
    video_path = 'C:/Users/raysu/Documents/src/argos_data/dump/2020_02_20_00267.avi'
    track_path = 'C:/Users/raysu/Documents/src/argos_data/dump/2020_02_20_00267.avi.h5'
    reviewer.setupReading(video_path, track_path, 'hdf')
    reviewer.gotoFrame(1)
    win = qw.QMainWindow()
    toolbar = win.addToolBar('Zoom')
    toolbar = win.addToolBar('Zoom')
    zi = qw.QAction('Zoom in')
    zi.triggered.connect(reviewer.left_view.zoomIn)
    zo = qw.QAction('Zoom out')
    zo.triggered.connect(reviewer.right_view.zoomOut)
    arena = qw.QAction('Select arena')
    arena.triggered.connect(reviewer.left_view.scene().setArenaMode)
    arena_reset = qw.QAction('Rset arena')
    arena_reset.triggered.connect(reviewer.right_view.scene().resetArena)
    roi = qw.QAction('Select rectangular ROIs')
    roi.triggered.connect(reviewer.left_view.scene().setRoiRectMode)
    poly = qw.QAction('Select polygon ROIs')
    poly.triggered.connect(reviewer.left_view.scene().setRoiPolygonMode)
    toolbar.addAction(zi)
    toolbar.addAction(zo)
    toolbar.addAction(arena)
    toolbar.addAction(roi)
    toolbar.addAction(poly)
    toolbar.addAction(arena_reset)
    toolbar.addAction(reviewer.tieViewsAction)
    toolbar.addAction(reviewer.autoColorAction)
    win.setCentralWidget(reviewer)
    win.show()
    # reviewer.before.setViewportRect(qc.QRectF(50, 50, 100, 100))
    sys.exit(app.exec_())


if __name__ == '__main__':
    # test_reviewwidget()
    test_review()
