# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-07-09 1:20 PM
"""Review and correct tracks"""

import sys
import os
from typing import List, Tuple, Union, Dict
import logging
import threading
from collections import defaultdict, OrderedDict
import numpy as np
import cv2
import pandas as pd
from PyQt5 import (
    QtWidgets as qw,
    QtCore as qc,
    QtGui as qg
)

from argos.constants import Change, DrawingGeom
from argos import utility as ut
from argos.utility import make_color, get_cmap_color, rect2points
from argos.frameview import FrameScene, FrameView
from argos.vreader import VideoReader
from argos.limitswidget import LimitsWidget


settings = ut.init()


class TrackReader(qc.QObject):
    """Class to read the tracking data"""
    sigEnd = qc.pyqtSignal()
    sigSavedFrames = qc.pyqtSignal(int)

    op_assign = 0
    op_swap = 1
    op_delete = 2

    def __init__(self, data_file):
        super(TrackReader, self).__init__()
        self.data_path = data_file
        if data_file.endswith('csv'):
            self.track_data = pd.read_csv(self.data_path)
        else:
            self.track_data = pd.read_hdf(self.data_path, 'tracked')
        self.track_data = self.track_data.astype({'frame': int, 'trackid': int})
        self.last_frame = self.track_data.frame.max()
        self.wmin = 0
        self.wmax = 1000
        self.hmin = 0
        self.hmax = 1000
        self.change_list = []

    @property
    def max_id(self):
        return self.track_data.trackid.max()

    @qc.pyqtSlot(int)
    def setWmin(self, val: int):
        self.wmin = val

    @qc.pyqtSlot(int)
    def setWmax(self, val: int):
        self.wmax = val

    @qc.pyqtSlot(int)
    def setHmin(self, val: int):
        self.hmin = val

    @qc.pyqtSlot(int)
    def setHmax(self, val: int):
        self.hmax = val

    def getTracks(self, frame_no):
        if frame_no > self.last_frame:
            self.sigEnd.emit()
            return
        self.frame_pos = frame_no
        tracks = self.track_data[self.track_data.frame == frame_no]
        # Filter bboxes violating size constraints
        wh = np.sort(tracks[['w', 'h']].values, axis=1)
        sel = np.flatnonzero((wh[:, 0] >= self.wmin) &
                             (wh[:, 0] <= self.wmax) &
                             (wh[:, 1] >= self.hmin) &
                             (wh[:, 0] <= self.hmax))
        tracks = tracks.iloc[sel]
        tracks = self.applyChanges(tracks)
        return tracks

    @qc.pyqtSlot(int, int, int)
    def changeTrack(self, frame_no, orig_id, new_id):
        """When user assigns `new_id` to `orig_id` keep it in undo buffer"""
        self.change_list.append(Change(frame=frame_no, change=self.op_assign,
                                       orig=orig_id, new=new_id))
        logging.debug(
            f'Changin track: frame: {frame_no}, old: {orig_id}, new: {new_id}')

    @qc.pyqtSlot(int, int, int)
    def swapTrack(self, frame_no, orig_id, new_id):
        """When user swaps `new_id` with `orig_id` keep it in swap buffer"""
        self.change_list.append(Change(frame=frame_no, change=self.op_swap,
                                       orig=orig_id, new=new_id))
        logging.debug(
            f'Swap track: frame: {frame_no}, old: {orig_id}, new: {new_id}')

    def deleteTrack(self, frame_no, orig_id):
        self.change_list.append(Change(frame=frame_no, change=self.op_delete,
                                       orig=orig_id, new=None))

    @qc.pyqtSlot(int)
    def undoChangeTrack(self, frame_no):
        for ii in range(len(self.change_list) - 1, 0, -1):
            if self.change_list[ii][0] < frame_no:
                break
        if ii == len(self.change_list):
            return
        self.change_list = self.change_list[:ii+1]

    def applyChanges(self, tdata):
        """Apply the changes in `change_list` to traks in `trackdf`
        `trackdf` should have a single `frame` value - changes  only
        upto and including this frame are applied.
        """
        if len(tdata) == 0:
            return {}
        tracks = {row.trackid: [row.x, row.y, row.w, row.h]
                  for row in tdata.itertuples()}
        frameno = tdata.frame.values[0]
        for change in self.change_list:
            if change.frame > frameno:
                break
            orig_trk = tracks.pop(change.orig, None)
            if change.change == self.op_swap:
                new_trk = tracks.pop(change.new, None)
                if orig_trk is not None:
                    tracks[change.new] = orig_trk
                if new_trk is not None:
                    tracks[change.orig] = new_trk
            elif change.change == self.op_assign:
                if orig_trk is not None:
                    tracks[change.new] = orig_trk
            elif change.change != self.op_delete:  # it must be delete, so leave the orig_trk popped
                raise ValueError(
                    f'Frame: {frameno}: Only alternative operation is '
                    f'`delete`({self.op_delete}) but found {change.change}')
        return tracks

    def saveChanges(self, filepath):
        """Consolidate all the changes made in track id assignment.

        Assumptions: as tracking progresses, only new, bigger numbers are
        assigned for track ids. track_id never goes down.

        track ids can be swapped.
        """
        # assignments = self.consolidateChanges()
        data = []

        for frame_no, tdata in self.track_data.groupby('frame'):
            tracks = self.applyChanges(tdata)
            for tid, tdata in tracks.items():
                data.append([frame_no, tid] + tdata)
                self.sigSavedFrames.emit(frame_no)
        data = pd.DataFrame(data=data,
                            columns=['frame', 'trackid', 'x', 'y', 'w', 'h'])
        if filepath.endswith('.csv'):
            data.to_csv(filepath, index=False)
        else:
            data.to_hdf(filepath, 'tracked', mode='w')
        self.track_data = data
        self.change_list = []



class ReviewScene(FrameScene):
    
    def __init__(self, *args, **kwargs):
        super(ReviewScene, self).__init__(*args, **kwargs)
        self.historic_track_ls = qc.Qt.DashLine
        self.hist_len = 1

    @qc.pyqtSlot(int)
    def setHistLen(self, age: int) -> None:
        self.hist_len = age

    @qc.pyqtSlot(dict)
    def setRectangles(self, rects: Dict[int, np.ndarray]) -> None:
        """rects: a dict of id: (x, y, w, h, flag)

        This overrides the same slot in FrameScene where each rectangle has
        a fifth entry indicating if this bbox/track is present in the current
        frame (1) or was scene in an earlier frame (0).

        The ones from earlier frame that are not present in the current frame
        are displayed with a special line style (default: dashes)
        """
        logging.debug(f'{self.objectName()} Received rectangles from {self.sender().objectName()}')
        logging.debug(f'{self.objectName()} Rectangles: {rects}')
        self.clearItems()
        logging.debug(f'{self.objectName()} cleared')

        for id_, tdata in rects.items():
            if tdata.shape[0] != 5:
                raise ValueError(f'Incorrectly sized entry: {id_}: {tdata}')
            if self.autocolor:
                color = qg.QColor(*make_color(id_))
            elif self.colormap is not None:
                color = qg.QColor(
                    *get_cmap_color(id_ % self.max_colors, self.max_colors,
                                    self.colormap))
            else:
                color = qg.QColor(self.color)
            # Use transparency to indicate age
            color.setAlpha(int(255 - 128.0 * tdata[4] / self.hist_len))
            pen = qg.QPen(color, self.linewidth)
            if tdata[4] > 0:
                pen.setStyle(self.historic_track_ls)
                logging.debug(f'{self.objectName()}: old track : {id_}')
            rect = tdata[:4].copy()
            item = self.addRect(*rect, pen)
            self.item_dict[id_] = item
            text = self.addText(str(id_), self.font)
            self.label_dict[id_] = text
            text.setDefaultTextColor(color)
            text.setPos(rect[0], rect[1])
            self.polygons[id_] = rect
            logging.debug(f'Set {id_}: {rect}')
        if self.arena is not None:
            self.addPolygon(self.arena, qg.QPen(qc.Qt.red))
        self.sigPolygons.emit(self.polygons)
        self.sigPolygonsSet.emit()



class TrackView(FrameView):
    """Visualization of bboxes of objects on video frame with facility to set
    visible area of scene"""
    sigSelected = qc.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(TrackView, self).__init__(*args, **kwargs)
        self.sigSelected.connect(self.scene().setSelected)

    def setViewportRect(self, rect: qc.QRectF) -> None:
        self.fitInView(rect.x(), rect.y(),
                       rect.width(),
                       rect.height(),
                       qc.Qt.KeepAspectRatio)

    def _makeScene(self):
        self.frame_scene = ReviewScene()
        self.setScene(self.frame_scene)


class TrackList(qw.QListWidget):
    sigMapTracks = qc.pyqtSignal(int, int, bool)
    sigSelected = qc.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(TrackList, self).__init__(*args, **kwargs)
        self._drag_button = qc.Qt.NoButton
        self.setSelectionMode(qw.QAbstractItemView.SingleSelection)
        self.itemSelectionChanged.connect(self.sendSelected)

    def decode_item_data(self, mime_data: qc.QMimeData) -> List[
        Dict[qc.Qt.ItemDataRole, qc.QVariant]]:
        """This was a test trick found here:
        https://wiki.python.org/moin/PyQt/Handling%20Qt%27s%20internal%20item%20MIME%20type
        but a much simpler solution for my case was here:
        https://stackoverflow.com/questions/9715171/how-to-drop-items-on-qlistwidget-between-some-items
        """
        data = mime_data.data('application/x-qabstractitemmodeldatalist')
        ds = qc.QDataStream(data)
        item = {}
        item_list = []
        while not ds.atEnd():
            row = ds.readInt32()
            col = ds.readInt32()
            map_items = ds.readInt32()
            for ii in range(map_items):
                key = ds.readInt32()
                value = qc.QVariant()
                ds >> value
                item[qc.Qt.ItemDataRole(key)] = value
            item_list.append(item)
        return item_list

    def dragMoveEvent(self, e: qg.QDragMoveEvent) -> None:
        """This is just for tracking left vs right mouse button drag"""
        self._drag_button = e.mouseButtons()
        super(TrackList, self).dragMoveEvent(e)

    def dropEvent(self, event: qg.QDropEvent) -> None:
        # items = self.decode_item_data(event.mimeData())
        # assert  len(items) == 1, 'Only allowed to drop a single item'
        # item = items[0]
        # logging.debug(f'data: {item[qc.Qt.DisplayRole].value()}')
        # If dragged with left button, rename. if right button, swap
        source = event.source().currentItem()
        target = self.itemAt(event.pos())
        if target is None:
            event.ignore()
            return
        self.sigMapTracks.emit(int(source.text()), int(target.text()),
                               self._drag_button == qc.Qt.RightButton )
        event.accept()

    @qc.pyqtSlot(list)
    def replaceAll(self, track_list: List[int]):
        """Replace all items with keys from new tracks dictionary"""
        self.clear()
        self.addItems([str(x) for x in sorted(track_list)])

    @qc.pyqtSlot()
    def sendSelected(self):
        items = [int(item.text()) for item in self.selectedItems()]
        self.sigSelected.emit(items)


class LimitWin(qw.QMainWindow):
    sigClose = qc.pyqtSignal(bool)  # connected to action checked state

    def __init__(self, *args, **kwargs):
        super(LimitWin, self).__init__(*args, **kwargs)

    def closeEvent(self, a0: qg.QCloseEvent) -> None:
        self.sigClose.emit(False)
        super(LimitWin, self).closeEvent(a0)


class ReviewWidget(qw.QWidget):
    """A widget with two panes for reviewing track mislabelings"""
    sigNextFrame = qc.pyqtSignal()
    sigGotoFrame = qc.pyqtSignal(int)
    sigLeftFrame = qc.pyqtSignal(np.ndarray, int)
    sigRightFrame = qc.pyqtSignal(np.ndarray, int)
    sigLeftTracks = qc.pyqtSignal(dict)
    sigLeftTrackList = qc.pyqtSignal(list)  # to separate tracks displayed on frame from those in list widget
    sigRightTracks = qc.pyqtSignal(dict)
    sigRightTrackList = qc.pyqtSignal(list)
    sigAllTracksList = qc.pyqtSignal(list)
    sigChangeTrack = qc.pyqtSignal(int, int, int)
    sigSetColormap = qc.pyqtSignal(str, int)
    sigDiffMessage = qc.pyqtSignal(str)
    sigUndoCurrentChanges = qc.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super(ReviewWidget, self).__init__(*args, **kwargs)
        # Keep track of all the tracks seen so far
        self.setObjectName('ReviewWidget')
        self.to_save = False
        self.history_length = 1
        self.all_tracks = OrderedDict()
        self.left_frame = None
        self.left_tracks = None
        self.right_frame = None
        self.right_tracks = None
        self.frame_no = -1
        self.speed = 1.0
        self.timer = qc.QTimer(self)
        self.timer.setSingleShot(True)
        self.video_reader = None
        self.track_reader = None
        self.left_tracks = {}
        self.right_tracks = {}
        self.roi = None
        # Since video seek is buggy, we have to do continuous reading
        self.left_frame = None
        self.right_frame = None
        layout = qw.QVBoxLayout()
        panes_layout = qw.QHBoxLayout()
        self.left_view = TrackView()
        self.left_view.setObjectName('LeftView')
        self.left_view.frame_scene.setObjectName('LeftScene')
        # self.left_view.setSizePolicy(qw.QSizePolicy.MinimumExpanding, qw.QSizePolicy.MinimumExpanding)
        self.left_view.setHorizontalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        self.left_view.setVerticalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        panes_layout.addWidget(self.left_view, 1)

        max_list_width = 100
        self.left_list = TrackList()
        self.left_list.setObjectName('LeftList')
        self.left_list.setMaximumWidth(max_list_width)
        # self.left_list.setSizePolicy(qw.QSizePolicy.Minimum, qw.QSizePolicy.Expanding)
        self.left_list.setDragEnabled(True)
        list_layout = qw.QVBoxLayout()
        list_layout.setSizeConstraint(qw.QLayout.SetMinimumSize)
        label = qw.QLabel('Left tracks')
        label.setSizePolicy(qw.QSizePolicy.Minimum, qw.QSizePolicy.Minimum)
        label.setMaximumWidth(max_list_width)
        list_layout.addWidget(label)
        list_layout.addWidget(self.left_list)
        panes_layout.addLayout(list_layout)
        self.all_list = TrackList()
        self.all_list.setMaximumWidth(max_list_width)
        self.all_list.setDragEnabled(True)
        self.all_list.setObjectName('AllList')
        list_layout = qw.QVBoxLayout()
        # list_layout.setSizeConstraint(qw.QLayout.SetMinimumSize)
        list_layout.addWidget(qw.QLabel('All tracks'))
        list_layout.addWidget(self.all_list)
        panes_layout.addLayout(list_layout)
        self.right_list = TrackList()
        self.right_list.setObjectName('RightList')
        self.right_list.setAcceptDrops(True)
        self.right_list.setMaximumWidth(max_list_width)
        list_layout = qw.QVBoxLayout()
        list_layout.setSizeConstraint(qw.QLayout.SetMinimumSize)
        list_layout.addWidget(qw.QLabel('Right tracks'))
        list_layout.addWidget(self.right_list)
        panes_layout.addLayout(list_layout)

        self.right_view = TrackView()
        self.right_view.setObjectName('RightView')
        self.right_view.frame_scene.setObjectName('RightScene')
        self.right_view.frame_scene.setArenaMode()
        # self.right_view.setSizePolicy(qw.QSizePolicy.Expanding,
        #                              qw.QSizePolicy.Expanding)
        self.right_view.setHorizontalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        self.right_view.setVerticalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        panes_layout.addWidget(self.right_view, 1)
        layout.addLayout(panes_layout)
        self.play_button = qw.QPushButton('Play')
        self.slider = qw.QSlider(qc.Qt.Horizontal)
        self.pos_spin = qw.QSpinBox()
        self.reset_button = qw.QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset)
        ctrl_layout = qw.QHBoxLayout()
        ctrl_layout.addWidget(self.play_button)
        ctrl_layout.addWidget(self.slider)
        ctrl_layout.addWidget(self.pos_spin)
        ctrl_layout.addWidget(self.reset_button)
        layout.addLayout(ctrl_layout)
        self.setLayout(layout)
        self.lim_widget = LimitsWidget(self)
        self.lim_win = LimitWin()
        self.lim_win.setCentralWidget(self.lim_widget)
        self.makeActions()
        self.makeShortcuts()
        self.timer.timeout.connect(self.nextFrame)
        self.sigLeftFrame.connect(self.left_view.setFrame)
        self.sigRightFrame.connect(self.right_view.setFrame)
        self.right_view.sigArena.connect(self.setRoi)
        self.sigLeftTracks.connect(self.left_view.sigSetRectangles)
        self.sigLeftTrackList.connect(self.left_list.replaceAll)
        self.sigRightTracks.connect(self.right_view.sigSetRectangles)
        self.sigRightTrackList.connect(self.right_list.replaceAll)
        self.sigAllTracksList.connect(self.all_list.replaceAll)
        self.left_list.sigSelected.connect(self.left_view.sigSelected)
        self.all_list.sigSelected.connect(self.left_view.sigSelected)
        self.right_list.sigSelected.connect(self.right_view.sigSelected)
        self.right_list.sigMapTracks.connect(self.mapTracks)
        self.play_button.clicked.connect(self.playVideo)
        self.play_button.setCheckable(True)
        self.slider.valueChanged.connect(self.gotoFrame)
        self.pos_spin.valueChanged.connect(self.gotoFrame)
        self.pos_spin.lineEdit().setEnabled(False)
        self.sigSetColormap.connect(self.left_view.frame_scene.setColormap)
        self.sigSetColormap.connect(self.right_view.frame_scene.setColormap)

    @qc.pyqtSlot()
    def setHistLen(self):
        val, ok = qw.QInputDialog.getInt(self, 'History length',
                                     'Oldest tracks to show (# of frames)',
                                     value=self.history_length,
                                     min=1)
        if ok:
            self.history_length = val
        self.left_view.frame_scene.setHistLen(val)
        self.right_view.frame_scene.setHistLen(val)

    @qc.pyqtSlot()
    def speedUp(self):
        self.speed *= 1.25
        logging.debug(f'Speed: {self.speed}')

    @qc.pyqtSlot()
    def slowDown(self):
        self.speed /= 1.25
        logging.debug(f'Speed: {self.speed}')

    @qc.pyqtSlot(bool)
    def tieViews(self, tie: bool) -> None:
        if tie:
            logging.debug(f'Before {self.left_view.viewport().size()}')
            logging.debug(f'After {self.right_view.viewport().size()}')
            self.left_view.sigViewportAreaChanged.connect(
                self.right_view.setViewportRect)
            self.right_view.sigViewportAreaChanged.connect(
                self.left_view.setViewportRect)
        else:
            try:
                self.left_view.sigViewportAreaChanged.disconnect()
            except TypeError:
                pass
            try:
                self.right_view.sigViewportAreaChanged.disconnect()
            except TypeError:
                pass

    def makeActions(self):
        self.disableSeekAction = qw.QAction('Disable seek')
        self.disableSeekAction.setCheckable(True)
        self.disableSeekAction.setChecked(True)
        self.disableSeekAction.triggered.connect(self.disableSeek)
        self.tieViewsAction = qw.QAction('Scroll views together')
        self.tieViewsAction.setCheckable(True)
        self.tieViewsAction.triggered.connect(self.tieViews)
        self.tieViewsAction.setChecked(True)
        self.tieViews(True)
        self.autoColorAction = qw.QAction('Automatic color')
        self.autoColorAction.setCheckable(True)
        self.autoColorAction.triggered.connect(
            self.left_view.autoColorAction.trigger)
        self.autoColorAction.triggered.connect(
            self.right_view.autoColorAction.trigger)
        self.autoColorAction.triggered.connect(self.setAutoColor)
        self.colormapAction = qw.QAction('Use colormap')
        self.colormapAction.triggered.connect(self.setColormap)
        self.colormapAction.setCheckable(True)
        self.setRoiAction = qw.QAction('Set polygon ROI')
        self.setRoiAction.triggered.connect(self.right_view.setArenaMode)
        self.right_view.resetArenaAction.triggered.connect(self.resetRoi)
        self.openAction = qw.QAction('Open tracked data')
        self.openAction.triggered.connect(self.openTrackedData)
        self.saveAction = qw.QAction('Save reviewed data')
        self.saveAction.triggered.connect(self.saveReviewedTracks)
        self.speedUpAction = qw.QAction('Double speed')
        self.speedUpAction.triggered.connect(self.speedUp)
        self.slowDownAction = qw.QAction('Half speed')
        self.slowDownAction.triggered.connect(self.slowDown)
        self.zoomInLeftAction = qw.QAction('Zoom-in left')
        self.zoomInLeftAction.triggered.connect(self.left_view.zoomIn)
        self.zoomInRightAction = qw.QAction('Zoom-in right')
        self.zoomInRightAction.triggered.connect(self.right_view.zoomIn)
        self.zoomOutLeftAction = qw.QAction('Zoom-out left')
        self.zoomOutLeftAction.triggered.connect(self.left_view.zoomOut)
        self.zoomOutRightAction = qw.QAction('Zoom-out right')
        self.zoomOutRightAction.triggered.connect(self.right_view.zoomOut)
        self.showOldTracksAction = qw.QAction('Show old tracks')
        self.showOldTracksAction.setCheckable(True)
        # self.showOldTracksAction.triggered.connect(self.all_list.setEnabled)
        self.playAction = qw.QAction('Play')
        self.playAction.triggered.connect(self.playVideo)
        self.resetAction = qw.QAction('Reset')
        self.resetAction.triggered.connect(self.reset)
        self.showDifferenceAction = qw.QAction('Show popup window for left/right mismatch')
        self.showDifferenceAction.setCheckable(True)
        show_difference = settings.value('review/showdiff', 1, type=int)
        self.showDifferenceAction.setChecked(show_difference)
        self.swapTracksAction = qw.QAction('Swap tracks')
        self.swapTracksAction.triggered.connect(self.swapTracks)
        self.replaceTrackAction = qw.QAction('Replace track')
        self.replaceTrackAction.triggered.connect(self.replaceTrack)
        self.deleteTrackAction = qw.QAction('Delete track')
        self.deleteTrackAction.triggered.connect(self.deleteSelected)
        self.undoCurrentChangesAction = qw.QAction('Undo changes in current frame')
        self.undoCurrentChangesAction.triggered.connect(self.undoCurrentChanges)
        self.showLimitsAction = qw.QAction('Size limits')
        self.showLimitsAction.setCheckable(True)
        self.showLimitsAction.setChecked(False)
        self.lim_win.setVisible(False)
        self.showLimitsAction.triggered.connect(self.lim_win.setVisible)
        self.lim_win.sigClose.connect(self.showLimitsAction.setChecked)
        self.histlenAction = qw.QAction('Set old track age limit')
        self.histlenAction.triggered.connect(self.setHistLen)

    def makeShortcuts(self):
        self.sc_play = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Space), self)
        self.sc_play.activated.connect(self.togglePlay)
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
        self.sc_remove.activated.connect(self.deleteSelected)
        self.sc_remove_2 = qw.QShortcut(qg.QKeySequence('X'), self)
        self.sc_remove_2.activated.connect(self.deleteSelected)
        self.sc_speedup = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Up), self)
        self.sc_speedup.activated.connect(self.speedUp)
        self.sc_slowdown = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Down), self)
        self.sc_slowdown.activated.connect(self.slowDown)
        self.sc_save = qw.QShortcut(qg.QKeySequence('Ctrl+S'), self)
        self.sc_save.activated.connect(self.saveReviewedTracks)
        self.sc_open = qw.QShortcut(qg.QKeySequence('Ctrl+O'), self)
        self.sc_open.activated.connect(self.openTrackedData)
        self.sc_undo = qw.QShortcut(qg.QKeySequence('Ctrl+Z'), self)
        self.sc_undo.activated.connect(self.undoCurrentChanges)

    @qc.pyqtSlot(bool)
    def disableSeek(self, checked: bool) -> None:
        if checked:
            try:
                self.sigGotoFrame.disconnect()
            except TypeError:
                pass
            if self.video_reader is not None:
                self.sigNextFrame.connect(self.video_reader.read)
        else:
            try:
                self.sigNextFrame.disconnect()
            except TypeError:
                pass
            if self.video_reader is not None:
                self.sigGotoFrame.connect(self.video_reader.gotoFrame)

    @qc.pyqtSlot()
    def deleteSelected(self) -> None:
        widget = qw.QApplication.focusWidget()
        if isinstance(widget, TrackList):
            items = widget.selectedItems()
        else:
            return
        selected = [int(item.text()) for item in items]
        self.right_view.scene().setSelected(selected)
        self.right_view.scene().removeSelected()
        for sel in selected:
            self.track_reader.deleteTrack(self.frame_no, sel)
            self.right_tracks.pop(sel)
            right_items = self.right_list.findItems(items[0].text(),
                                                    qc.Qt.MatchExactly)
            for item in right_items:
                self.right_list.takeItem(self.right_list.row(item))
        self.to_save = True

    @qc.pyqtSlot()
    def swapTracks(self):
        source = self.all_list.selectedItems()
        target = self.right_list.selectedItems()
        if len(source) == 0 or len(target) == 0:
            return
        self.mapTracks(int(source[0].text()), int(target[0].text()),
                               True)

    @qc.pyqtSlot()
    def replaceTrack(self):
        source = self.all_list.selectedItems()
        target = self.right_list.selectedItems()
        if len(source) == 0 or len(target) == 0:
            return
        self.mapTracks(int(source[0].text()), int(target[0].text()), False)

    @qc.pyqtSlot(int)
    def gotoFrame(self, frame_no):
        if self.track_reader is None or \
                self.video_reader is None or \
                frame_no >= self.track_reader.last_frame:
            return
        if self.disableSeekAction.isChecked():
            self.sigNextFrame.emit()
        else:
            self.frame_no = frame_no
            logging.debug(f'Frame no set: {frame_no}')
            if self.frame_no > 0:
                self.sigGotoFrame.emit(self.frame_no - 1)
            self.sigGotoFrame.emit(self.frame_no)

    @qc.pyqtSlot()
    def gotoEditedPos(self):
        frame_no = int(self.pos_spin.text())
        self.gotoFrame(frame_no)

    @qc.pyqtSlot()
    def nextFrame(self):
        self.gotoFrame(self.frame_no + 1)
        if self.play_button.isChecked():
            self.timer.start(self.frame_interval / self.speed)

    @qc.pyqtSlot()
    def prevFrame(self):
        if self.frame_no > 0:
            self.gotoFrame(self.frame_no - 1)

    def _flag_tracks(self, all_tracks, cur_tracks):
        """Change the track info to include it age, if age is more than
        `history_length` then remove this track from all tracks -
        this avoids cluttering the view with very old tracks that have not been
        seen in a long time."""
        pop = []
        for tid, rect in all_tracks.items():
            rect[4] += 1
            if rect[4] > self.history_length:
               pop.append(tid)
        [all_tracks.pop(tid) for tid in pop]
        for tid, rect in cur_tracks.items():
            all_tracks[tid] = np.r_[rect[:4], 0]
        return all_tracks

    @qc.pyqtSlot(qg.QPolygonF)
    def setRoi(self, roi: qg.QPolygonF) -> None:
        self.roi = roi

    @qc.pyqtSlot()
    def resetRoi(self):
        self.roi = None

    @qc.pyqtSlot()
    def undoCurrentChanges(self):
        self.sigUndoCurrentChanges.emit(self.frame_no)

    @qc.pyqtSlot(np.ndarray, int)
    def setFrame(self, frame: np.ndarray, pos: int) -> None:
        logging.debug(f'Received frame: {pos}')
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self.pos_spin.blockSignals(True)
        self.pos_spin.setValue(pos)
        self.pos_spin.blockSignals(False)
        tracks = self.track_reader.getTracks(pos)
        if self.roi is not None:
            # flag tracks outside ROI
            include = {}
            track_ids = list(tracks.keys())
            for tid in track_ids:
                vertices = rect2points(np.array(tracks[tid]))            
                contained = [self.roi.containsPoint(qc.QPointF(*vtx), qc.Qt.OddEvenFill)
                             for vtx in vertices]
                if not np.any(contained):
                    self.track_reader.deleteTrack(self.frame_no, tid)
                    tracks.pop(tid)                    
                
        old_all_tracks = self.all_tracks.copy()
        self._flag_tracks(self.all_tracks, tracks)
        self.sigAllTracksList.emit(list(self.all_tracks.keys()))
        if self.disableSeekAction.isChecked():
            self.frame_no = pos
            self.left_frame = self.right_frame
            self.right_frame = frame
            self.left_tracks = self.right_tracks
            self.right_tracks = self._flag_tracks({}, tracks)
            if self.left_frame is not None:
                self.sigLeftFrame.emit(self.left_frame, pos - 1)
            self.sigRightFrame.emit(self.right_frame, pos)
            if self.showOldTracksAction.isChecked():
                self.sigLeftTracks.emit(old_all_tracks)
                self.sigRightTracks.emit(self.all_tracks)
            else:
                self.sigLeftTracks.emit(self.left_tracks)
                self.sigRightTracks.emit(self.right_tracks)
            self.sigLeftTrackList.emit(list(self.left_tracks.keys()))
            self.sigRightTrackList.emit(list(self.right_tracks.keys()))
        elif pos == self.frame_no - 1:
            logging.debug(f'Received left frame: {pos}')
            self.sigLeftFrame.emit(frame, pos)
            self.left_tracks = self._flag_tracks({}, tracks)
            if self.showOldTracksAction.isChecked():
                self.sigLeftTracks.emit(self.all_tracks)
            else:
                self.sigLeftTracks.emit(self.left_tracks)
            self.sigLeftTrackList.emit(list(self.left_tracks.keys()))
            self._wait_cond.set()
            return  # do not show popup message for old frame
        elif pos == self.frame_no:
            logging.debug(f'right frame: {pos}')
            self.sigRightFrame.emit(frame, pos)
            self.right_tracks = self._flag_tracks({}, tracks)
            if self.showOldTracksAction.isChecked():
                self.sigRightTracks.emit(self.all_tracks)
            else:
                self.sigRightTracks.emit(self.right_tracks)
            self.sigRightTrackList.emit(list(self.right_tracks.keys()))
            # Pause if there is a mismatch with the earlier tracks
        else:
            raise Exception('This should not be reached')
        message = self._get_diff()
        if len(message) > 0:
            if self.showDifferenceAction.isChecked():
                qw.QMessageBox.information(self, 'Track mismatch', message)
            self.sigDiffMessage.emit(message)
        self._wait_cond.set()
        logging.debug('wait condition set')

    def _get_diff(self):
        left_keys = set(self.left_tracks.keys())
        right_keys = set(self.right_tracks.keys())
        if left_keys != right_keys:
            self.play_button.setChecked(False)
            self.playVideo(False)
            logging.info(f'Tracks don\'t match between frames {self.frame_no} '
                         f'and {self.frame_no + 1}: '
                         f'{left_keys.symmetric_difference(right_keys)}')
            left_only = left_keys - right_keys
            left_message = f'Tracks only on left: {left_only}.' \
                if len(left_only) > 0 else ''
            right_only = right_keys - left_keys
            right_message = f'Tracks only on right: {right_only}.' \
                if len(right_only) > 0 else ''
            return f'Frame {self.frame_no}-{self.frame_no+1}: {left_message} {right_message}'
        else:
            return ''

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
        fmt = 'csv' if filter.startswith('Text') else 'hdf'
        self.setupReading(vid_filename, track_filename)
        self.video_filename = vid_filename
        self.track_filename = track_filename
        settings.setValue('data/directory', os.path.dirname(track_filename))
        settings.setValue('video/directory', os.path.dirname(vid_filename))
        self.all_tracks.clear()
        self.left_list.clear()
        self.right_list.clear()
        self.left_tracks = {}
        self.right_tracks = {}
        self.left_frame = None
        self.right_frame = None
        self.gotoFrame(0)
        self.updateGeometry()

    def setupReading(self, video_path, data_path):
        self._wait_cond = threading.Event()
        try:
            self.video_reader = VideoReader(video_path, self._wait_cond)
        except IOError:
            return
        self.all_tracks.clear()
        self.right_view.resetArenaAction.trigger()
        self.track_reader = TrackReader(data_path)
        self.history_length = self.track_reader.last_frame
        self.lim_widget.sigWmin.connect(self.track_reader.setWmin)
        self.lim_widget.sigWmax.connect(self.track_reader.setWmax)
        self.lim_widget.sigHmin.connect(self.track_reader.setHmin)
        self.lim_widget.sigHmax.connect(self.track_reader.setHmax)
        if self.disableSeekAction.isChecked():
            self.sigNextFrame.connect(self.video_reader.read)
        else:
            self.sigGotoFrame.connect(self.video_reader.gotoFrame)        
        self.video_reader.sigFrameRead.connect(self.setFrame)
        self.frame_interval = 1000.0 / self.video_reader.fps
        self.pos_spin.setRange(0, self.track_reader.last_frame)
        self.slider.setRange(0, self.track_reader.last_frame)
        self.sigChangeTrack.connect(self.track_reader.changeTrack)
        self.sigUndoCurrentChanges.connect(self.track_reader.undoChangeTrack)

    @qc.pyqtSlot()
    def saveReviewedTracks(self):
        datadir = settings.value('data/directory', '.')
        track_filename, filter = qw.QFileDialog.getSaveFileName(
            self,
            'Save reviewed data',
            datadir, filter='HDF5 (*.h5 *.hdf);; Text (*.csv)')
        logging.debug(f'filename:{track_filename}\nselected filter:{filter}')
        if len(track_filename) > 0:
            try:
                indicator = qw.QProgressDialog('Saving track data', None,
                                               0,
                                               self.track_reader.last_frame + 1,
                                               self)

                indicator.setWindowModality(qc.Qt.WindowModal)
                indicator.show()
                self.track_reader.sigSavedFrames.connect(indicator.setValue)
                self.track_reader.saveChanges(track_filename)
                indicator.setValue(self.track_reader.last_frame + 1)
                self.to_save = False
            except OSError as err:
                qw.QMessageBox.critical(
                    self, 'Error opening file for writing',
                    f'File {track_filename} could not be opened.\n{err}')

    @qc.pyqtSlot()
    def doQuit(self):
        # self._wait_cond.set()
        if self.to_save:
            self.saveReviewedTracks()
        settings.setValue('review/showdiff', int(self.showDifferenceAction.isChecked()))

    @qc.pyqtSlot(bool)
    def playVideo(self, play: bool):
        if play:
            self.play_button.setText('Pause')
            self.playAction.setText('Pause')
            self.timer.start(self.frame_interval / self.speed)
        else:
            self.play_button.setText('Play')
            self.playAction.setText('Play')
            self.timer.stop()

    @qc.pyqtSlot()
    def togglePlay(self):
        if self.play_button.isChecked():
            self.play_button.setChecked(False)
            self.playVideo(False)
        else:
            self.play_button.setChecked(True)
            self.playVideo(True)

    @qc.pyqtSlot()
    def reset(self):
        """Reset video: reopen video and track file"""
        if self.video_reader is None:
            # Not initialized - do nothing
            return
        self._wait_cond.set()
        self.playVideo(False)
        self.left_view.clearAll()
        self.right_view.clearAll()
        self.all_tracks.clear()
        self.setupReading(self.video_filename, self.track_filename)
        self.gotoFrame(0)

    @qc.pyqtSlot(int, int, bool)
    def mapTracks(self, cur: int, tgt: int, swap: bool) -> None:
        if swap:
            self.track_reader.swapTrack(self.frame_no, tgt, cur)
        else:
            self.track_reader.changeTrack(self.frame_no, tgt, cur)
        tracks = self.track_reader.getTracks(self.frame_no)
        self.sigRightTrackList.emit(list(tracks.keys()))
        self.right_tracks = self._flag_tracks({}, tracks)
        self.sigRightTracks.emit(self.right_tracks)
        self.to_save = True

    @qc.pyqtSlot(bool)
    def setColormap(self, checked):
        if not checked:
            return
        input, accept = qw.QInputDialog.getItem(self, 'Select colormap',
                                                'Colormap',
                                                ['jet',
                                                 'viridis',
                                                 'rainbow',
                                                 'autumn',
                                                 'summer',
                                                 'winter',
                                                 'spring',
                                                 'cool',
                                                 'hot',
                                                 'None'])
        logging.debug(f'Setting colormap to {input}')
        if input == 'None':
            self.colormapAction.setChecked(False)
            return
        if not accept:
            return
        max_colors, accept = qw.QInputDialog.getInt(self, 'Number of colors',
                                                    'Number of colors', 10, 1,
                                                    20)
        if not accept:
            return
        self.autoColorAction.setChecked(False)
        self.colormapAction.setChecked(True)
        self.sigSetColormap.emit(input, max_colors)

    @qc.pyqtSlot(bool)
    def setAutoColor(self, checked):
        if checked:
            self.colormapAction.setChecked(False)


class ReviewerMain(qw.QMainWindow):
    sigQuit = qc.pyqtSignal()

    def __init__(self):
        super(ReviewerMain, self).__init__()
        self.review_widget = ReviewWidget()
        file_menu = self.menuBar().addMenu('&File')
        file_menu.addAction(self.review_widget.openAction)
        file_menu.addAction(self.review_widget.saveAction)
        view_menu = self.menuBar().addMenu('&View')
        view_menu.addAction(self.review_widget.zoomInLeftAction)
        view_menu.addAction(self.review_widget.zoomInRightAction)
        view_menu.addAction(self.review_widget.zoomOutLeftAction)
        view_menu.addAction(self.review_widget.zoomOutRightAction)
        view_menu.addAction(self.review_widget.tieViewsAction)
        view_menu.addAction(self.review_widget.autoColorAction)
        view_menu.addAction(self.review_widget.colormapAction)
        view_menu.addAction(self.review_widget.showOldTracksAction)
        view_menu.addAction(self.review_widget.showDifferenceAction)
        view_menu.addAction(self.review_widget.showLimitsAction)
        view_menu.addAction(self.review_widget.histlenAction)
        play_menu = self.menuBar().addMenu('Play')
        play_menu.addAction(self.review_widget.disableSeekAction)
        play_menu.addAction(self.review_widget.playAction)
        play_menu.addAction(self.review_widget.speedUpAction)
        play_menu.addAction(self.review_widget.slowDownAction)
        play_menu.addAction(self.review_widget.resetAction)
        action_menu = self.menuBar().addMenu('Action')
        action_menu.addActions([self.review_widget.swapTracksAction,
                                self.review_widget.replaceTrackAction,
                                self.review_widget.deleteTrackAction,
                                self.review_widget.undoCurrentChangesAction,
                                self.review_widget.right_view.resetArenaAction])
        self.debugAction = qw.QAction('Debug')
        self.debugAction.setCheckable(True)
        v = settings.value('review/debug', False, type=bool)
        self.debugAction.setChecked(v)
        self.debugAction.triggered.connect(self.setDebug)
        action_menu.addAction(self.debugAction)
        toolbar = self.addToolBar('View')
        toolbar.addActions(view_menu.actions())
        toolbar.addActions(action_menu.actions())
        self.setCentralWidget(self.review_widget)
        self.sigQuit.connect(self.review_widget.doQuit)
        self.status_label = qw.QLabel()
        self.review_widget.sigDiffMessage.connect(self.status_label.setText)
        self.statusBar().addWidget(self.status_label)

    @qc.pyqtSlot(bool)
    def setDebug(self, val: bool):
        settings.setValue('review/debug', val)
        if val:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

    @qc.pyqtSlot()
    def cleanup(self):
        self.sigQuit.emit()
        settings.sync()
        logging.debug('Saved settings')



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
    video_path = 'C:/Users/raysu/Documents/src/argos_data/dump/2020_02_20_00270.avi'
    track_path = 'C:/Users/raysu/Documents/src/argos_data/dump/2020_02_20_00270.avi.track.csv'
    reviewer.setupReading(video_path, track_path)
    reviewer.gotoFrame(0)
    win = qw.QMainWindow()
    toolbar = win.addToolBar('Zoom')
    zi = qw.QAction('Zoom in')
    zi.triggered.connect(reviewer.left_view.zoomIn)
    zo = qw.QAction('Zoom out')
    zo.triggered.connect(reviewer.right_view.zoomOut)
    arena = qw.QAction('Select arena')
    arena.triggered.connect(reviewer.left_view.scene().setArenaMode)
    arena_reset = qw.QAction('Reset arena')
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
    toolbar.addAction(reviewer.colormapAction)
    toolbar.addAction(reviewer.showOldTracksAction)
    win.setCentralWidget(reviewer)
    win.show()
    # reviewer.before.setViewportRect(qc.QRectF(50, 50, 100, 100))
    sys.exit(app.exec_())


if __name__ == '__main__':
    # test_reviewwidget()
    # test_review()
    app = qw.QApplication(sys.argv)
    win = ReviewerMain()
    win.setMinimumSize(800, 600)
    win.setWindowTitle('Argos - review tracks')
    win.showMaximized()
    app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())
