# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-07-09 1:20 PM
"""Review and correct tracks"""

import sys
import os
import csv
from queue import PriorityQueue
from typing import List, Tuple, Union, Dict
import logging
import threading
from collections import defaultdict, OrderedDict
from operator import attrgetter
import numpy as np
import cv2
import pandas as pd
from sortedcontainers import SortedKeyList
from PyQt5 import (
    QtWidgets as qw,
    QtCore as qc,
    QtGui as qg
)

from argos.constants import Change
from argos import utility as ut
from argos.utility import make_color, get_cmap_color, rect2points
from argos.frameview import FrameScene, FrameView
from argos.vreader import VideoReader
from argos.limitswidget import LimitsWidget
from argos.vwidget import VidInfo


settings = ut.init()


class TrackReader(qc.QObject):
    """Class to read the tracking data"""
    sigEnd = qc.pyqtSignal()
    sigSavedFrames = qc.pyqtSignal(int)
    sigChangeList = qc.pyqtSignal(SortedKeyList)

    op_assign = 0
    op_swap = 1
    op_delete = 2

    change_code = {op_assign: 'assign',
                   op_swap: 'swap',
                   op_delete: 'delete'}

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
        self.change_list = SortedKeyList(key=attrgetter('frame'))
        self.undone_changes = set()

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

    def getTrackId(self, track_id, frame_no=None, hist=10):
        """Get all entries for a track across frames.
        frame_no: if specified, only entries around this frame are returned.
        hist: at most these many past and future entries around `frame_no` are returned.
        """
        track = self.track_data[self.track_data.trackid == track_id].copy()
        if frame_no is None:
            return track
        tgt = track[track.frame == frame_no]
        if len(tgt) == 0:
            return None
        pos = track.index.get_loc(tgt.iloc[0].name)
        pre = max(0, pos - hist)
        post = min(len(track), pos + hist)
        return track.iloc[pre: post].copy()
        

    def getTracks(self, frame_no):
        if frame_no > self.last_frame:
            logging.debug(f'Reached last frame with tracks: frame no {frame_no}')
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
        change = Change(frame=frame_no, change=self.op_assign,
                                       orig=orig_id, new=new_id)
        self.change_list.add(change)
        self.sigChangeList.emit(self.change_list)
        logging.debug(
            f'Changin track: frame: {frame_no}, old: {orig_id}, new: {new_id}')

    @qc.pyqtSlot(int, int, int)
    def swapTrack(self, frame_no, orig_id, new_id):
        """When user swaps `new_id` with `orig_id` keep it in swap buffer"""
        change = Change(frame=frame_no, change=self.op_swap,
                                       orig=orig_id, new=new_id)
        self.change_list.add(change)
        logging.debug(
            f'Swap track: frame: {frame_no}, old: {orig_id}, new: {new_id}')

    def deleteTrack(self, frame_no, orig_id):
        change = Change(frame=frame_no, change=self.op_delete,
                                       orig=orig_id, new=None)
        self.change_list.add(change)

    @qc.pyqtSlot(int)
    def undoChangeTrack(self, frame_no):
        """This puts the specified frame in a blacklist so all changes applied
        on it are ignored"""
        while True:
            loc = self.change_list.bisect_key_left(frame_no)
            if loc < len(self.change_list) and \
               self.change_list[loc].frame == frame_no:
                self.change_list.pop(loc)
            else:
                return            

    def applyChanges(self, tdata):
        """Apply the changes in `change_list` to traks in `trackdf`
        `trackdf` should have a single `frame` value - changes  only
        upto and including this frame are applied.
        """
        if len(tdata) == 0:
            return {}
        tracks = {row.trackid: [row.x, row.y, row.w, row.h, row.frame]
                  for row in tdata.itertuples()}
        frameno = tdata.frame.values[0]
        for change in self.change_list:
            if change.frame > frameno:
                break
            if change.frame in self.undone_changes:
                continue
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
                data.append([frame_no, tid] + tdata[:4])
                qw.QApplication.processEvents()
            self.sigSavedFrames.emit(frame_no)
        data = pd.DataFrame(data=data,
                            columns=['frame', 'trackid', 'x', 'y', 'w', 'h'])
        if filepath.endswith('.csv'):
            data.to_csv(filepath, index=False)
        else:
            data.to_hdf(filepath, 'tracked', mode='w')
            
        self.track_data = data
        self.change_list.clear()
        self.sigChangeList.emit(self.change_list)

    @qc.pyqtSlot(str)
    def saveChangeList(self, fname: str) -> None:
        # self.change_list = sorted(self.change_list, key=attrgetter('frame'))
        with open(fname, 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(['frame', 'change', 'old', 'new'])
            for change in self.change_list:
                if change.frame not in self.undone_changes:
                    writer.writerow([change.frame, change.change, change.orig,
                                 change.new])

    @qc.pyqtSlot(str)
    def loadChangeList(self, fname: str) -> None:
        self.change_list.clear()
        with open(fname) as fd:
            first = True
            reader = csv.reader(fd)
            for row in reader:
                if not first and len(row) > 0:
                    new = int(row[3]) if len(row[3]) > 0 else None
                    change = Change(frame=int(row[0]), change=int(row[1]),
                               orig=int(row[2]), new=new)
                    self.change_list.add(change)
                first = False
        self.sigChangeList.emit(self.change_list)


class ReviewScene(FrameScene):
    
    def __init__(self, *args, **kwargs):
        super(ReviewScene, self).__init__(*args, **kwargs)
        self.historic_track_ls = qc.Qt.DashLine
        self.hist_gradient = 1
        self.track_hist = []

    @qc.pyqtSlot(int)
    def setHistGradient(self, age: int) -> None:
        self.hist_gradient = age

    @qc.pyqtSlot(np.ndarray)
    def showTrackHist(self, track: np.ndarray) -> None:
        for item in self.track_hist:
            self.removeItem(item)
        self.track_hist = []
        for ii, t in enumerate(track):
            color = qg.QColor(*get_cmap_color(ii, len(track), 'viridis'))
            self.track_hist.append(self.addEllipse(t[0] - 1, t[1] - 1, 2, 2, qg.QPen(color)))
        # self.track_hist = [self.addEllipse(t[0] - 1, t[1] - 1, 2, 2, qg.QPen(self.selected_color)) for t in track]
    
    @qc.pyqtSlot(dict)
    def setRectangles(self, rects: Dict[int, np.ndarray]) -> None:
        """rects: a dict of id: (x, y, w, h, frame)

        This overrides the same slot in FrameScene where each rectangle has
        a fifth entry indicating frame no of the rectangle.

        The ones from earlier frame that are not present in the current frame
        are displayed with a special line style (default: dashes)
        """
        logging.debug(f'{self.objectName()} Received rectangles from {self.sender().objectName()}')
        logging.debug(f'{self.objectName()} Rectangles: {rects}')
        self.clearItems()
        self.track_hist = []
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
            color.setAlpha(int(255 * (1 - 0.9 * min(np.abs(self.frameno - tdata[4]), self.hist_gradient) / self.hist_gradient)))
            pen = qg.QPen(color, self.linewidth)
            if tdata[4] != self.frameno:
                pen.setStyle(self.historic_track_ls)
                logging.debug(f'{self.objectName()}: old track : {id_}')
            rect = tdata[:4].copy()
            item = self.addRect(*rect, pen)
            self.item_dict[id_] = item
            text = self.addText(str(id_), self.font)
            self.label_dict[id_] = text
            text.setDefaultTextColor(color)
            text.setPos(rect[0], rect[1] - text.boundingRect().height())
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
        """Intermediate slot to convert text labels into integer track ids"""
        items = [int(item.text()) for item in self.selectedItems()]
        self.sigSelected.emit(items)


class LimitWin(qw.QMainWindow):
    sigClose = qc.pyqtSignal(bool)  # connected to action checked state

    def __init__(self, *args, **kwargs):
        super(LimitWin, self).__init__(*args, **kwargs)

    def closeEvent(self, a0: qg.QCloseEvent) -> None:
        self.sigClose.emit(False)
        super(LimitWin, self).closeEvent(a0)


class ChangeWindow(qw.QMainWindow):
    cols = ['frame', 'change', 'old id', 'new id']
    def __init__(self):
        super(ChangeWindow, self).__init__()
        self.table = qw.QTableWidget()
        self.table.setColumnCount(len(self.cols))
        self.table.setHorizontalHeaderLabels(self.cols)
        self.setCentralWidget(self.table)

    @qc.pyqtSlot(SortedKeyList)
    def setChangeList(self, change_list):
        self.table.clearContents()
        self.table.setRowCount(len(change_list))
        for ii, change in enumerate(change_list):
            self.table.setItem(ii, 0, qw.QTableWidgetItem(str(change.frame)))
            self.table.setItem(ii, 1, qw.QTableWidgetItem(
                TrackReader.change_code[change.change]))
            self.table.setItem(ii, 2, qw.QTableWidgetItem(str(change.orig)))
            self.table.setItem(ii, 3, qw.QTableWidgetItem(str(change.new)))


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
    sigDataFile = qc.pyqtSignal(str)
    sigProjectTrackHist = qc.pyqtSignal(np.ndarray)
    sigProjectTrackHistAll = qc.pyqtSignal(np.ndarray)

    def __init__(self, *args, **kwargs):
        super(ReviewWidget, self).__init__(*args, **kwargs)
        # Keep track of all the tracks seen so far
        self.setObjectName('ReviewWidget')
        self._wait_cond = threading.Event()
        self.breakpoint = -1
        self.entry_break = -1
        self.exit_break = -1
        self.history_length = 1
        self.all_tracks = OrderedDict()
        self.left_frame = None
        self.right_frame = None
        self.right_tracks = None
        self.frame_no = -1
        self.speed = 1.0
        self.timer = qc.QTimer(self)
        self.timer.setSingleShot(True)
        self.video_reader = None
        self.track_reader = None
        self.vid_info = VidInfo()
        self.left_tracks = {}
        self.right_tracks = {}
        self.roi = None
        # Since video seek is buggy, we have to do continuous reading
        self.left_frame = None
        self.right_frame = None
        self.save_indicator = None
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
        self.changelist_widget = ChangeWindow()
        self.changelist_widget.setVisible(False)
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
        self.all_list.sigSelected.connect(self.projectTrackHist)
        self.all_list.sigSelected.connect(self.left_view.sigSelected)
        self.right_list.sigSelected.connect(self.right_view.sigSelected)
        self.right_list.sigSelected.connect(self.projectTrackHist)
        self.sigProjectTrackHist.connect(self.right_view.frame_scene.showTrackHist)
        self.sigProjectTrackHistAll.connect(self.left_view.frame_scene.showTrackHist)
        self.right_list.sigMapTracks.connect(self.mapTracks)
        self.play_button.clicked.connect(self.playVideo)
        self.play_button.setCheckable(True)
        self.slider.valueChanged.connect(self.gotoFrame)
        self.pos_spin.valueChanged.connect(self.gotoFrame)
        self.pos_spin.lineEdit().setEnabled(False)
        self.right_view.sigSetColormap.connect(self.left_view.frame_scene.setColormap)
        # self.sigSetColormap.connect(self.left_view.frame_scene.setColormap)
        # self.sigSetColormap.connect(self.right_view.frame_scene.setColormap)

    @qc.pyqtSlot(list)
    def projectTrackHist(self, selected: list) -> None:
        if not self.showHistoryAction.isChecked():
            return
        
        for sel in  selected:
            if self.sender() == self.right_list:
                track = self.track_reader.getTrackId(sel, self.frame_no, self.history_length)
            else:
                track = self.track_reader.getTrackId(sel, None)
            if track is None:
                return
            track.loc[:, 'x'] += track.w / 2.0
            track.loc[:, 'y'] += track.h / 2.0
            if self.sender() == self.right_list:
                self.sigProjectTrackHist.emit(track[['x', 'y']].values)
            else:
                self.sigProjectTrackHistAll.emit(track[['x', 'y']].values)

    @qc.pyqtSlot(Exception)
    def catchSeekError(self, err: Exception)-> None:
        qw.QMessageBox.critical(self, 'Error jumping frames', str(err))
        self.disableSeek(True)
        self.pos_spin.lineEdit().setEnabled(False)
        self.disableSeekAction.setChecked(True)

    @qc.pyqtSlot()
    def setBreakpoint(self):
        val, ok = qw.QInputDialog.getInt(self, 'Set breakpoint',
                                         'Pause at frame #',
                                         value=self.breakpoint,
                                         min=0)
        if ok:
            self.breakpoint = val

    @qc.pyqtSlot()
    def clearBreakpoint(self):
        if self.video_reader is not None:
            self.breakpoint = self.video_reader.frame_count

    @qc.pyqtSlot()
    def setBreakpointAtCurrent(self):
        self.breakpoint = self.frame_no

    @qc.pyqtSlot()
    def setBreakpointAtEntry(self):
        val, ok = qw.QInputDialog.getInt(self, 'Set breakpoint at entry',
                                         'Pause at appearance of trackid #',
                                         value=-1,
                                         min=-1)
        if ok:
            self.entry_break = val

    @qc.pyqtSlot()
    def setBreakpointAtExit(self):
        val, ok = qw.QInputDialog.getInt(self, 'Set breakpoint on exit',
                                         'Pause at disappearance of trackid #',
                                         value=-1,
                                         min=-1)
        if ok:
            self.exit_break = val

    @qc.pyqtSlot()
    def clearBreakpointAtEntry(self):
        self.entry_break = -1

    @qc.pyqtSlot()
    def clearBreakpointAtExit(self):
        self.exit_break = -1
        
    def breakpointMessage(self, pos):
        if pos == self.breakpoint:
            self.play_button.setChecked(False)
            self.playVideo(False)
            qw.QMessageBox.information(
                self, 'Processing paused',
                f'Reached breakpoint at frame # {self.breakpoint}')

    def entryExitMessage(self, left, right):
        do_break = False
        if (self.entry_break in right) and (self.entry_break not in left):
            do_break = True
            message = f'Reached breakpoint at entry of # {self.entry_break}'
        elif (self.exit_break in left) and (self.exit_break not in right):
            do_break = True
            message = f'Reached breakpoint at exit of # {self.exit_break}'
        if do_break:
            self.play_button.setChecked(False)
            self.playVideo(False)
            qw.QMessageBox.information(
                self, 'Processing paused',
                message)

    @qc.pyqtSlot()
    def setHistLen(self):
        val, ok = qw.QInputDialog.getInt(self, 'History length',
                                     'Oldest tracks to keep (# of frames)',
                                     value=self.history_length,
                                     min=1)
        if ok:
            self.history_length = val

    @qc.pyqtSlot()
    def setHistGradient(self):
        val, ok = qw.QInputDialog.getInt(
            self, 'Color-gradient for  old tracks',
            'Faintest of old tracks (# of frames)',
            value=self.right_view.frame_scene.hist_gradient,
            min=1)
        if ok:
            self.left_view.frame_scene.setHistGradient(val)
            self.right_view.frame_scene.setHistGradient(val)

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
            self.right_view.horizontalScrollBar().valueChanged.connect(
                self.left_view.horizontalScrollBar().setValue)
            self.right_view.verticalScrollBar().valueChanged.connect(
                self.left_view.verticalScrollBar().setValue)
        else:
            try:
                self.left_view.sigViewportAreaChanged.disconnect()
            except TypeError:
                pass
            try:
                self.right_view.sigViewportAreaChanged.disconnect()
            except TypeError:
                pass
            try:
                self.right_view.horizontalScrollBar().valueChanged.disconnect(
                  self.left_view.horizontalScrollBar().setValue)
            except TypeError:
                pass
            try:
                self.right_view.verticalScrollBar().valueChanged.disconnect(
                    self.left_view.verticalScrollBar().setValue)
            except TypeError:
                pass

    def makeActions(self):
        self.disableSeekAction = qw.QAction('Disable seek')
        self.disableSeekAction.setCheckable(True)
        disable_seek = settings.value('review/disable_seek', True, type=bool)
        self.disableSeek(disable_seek)
        self.disableSeekAction.setChecked(disable_seek)
        self.disableSeekAction.triggered.connect(self.disableSeek)
        self.tieViewsAction = qw.QAction('Scroll views together')
        self.tieViewsAction.setCheckable(True)
        self.tieViewsAction.triggered.connect(self.tieViews)
        self.tieViewsAction.setChecked(True)
        self.tieViews(True)
        self.autoColorAction = self.right_view.autoColorAction
        self.autoColorAction.triggered.connect(
            self.left_view.autoColorAction.trigger)
        # self.autoColorAction.triggered.connect(self.left_view.autoColorAction)
        self.colormapAction = self.right_view.colormapAction
        # self.colormapAction = qw.QAction('Colormap')
        # self.colormapAction.triggered.connect(self.setColormap)
        self.lineWidthAction = self.right_view.lineWidthAction
        self.right_view.sigLineWidth.connect(self.left_view.frame_scene.setLineWidth)
        self.setRoiAction = qw.QAction('Set polygon ROI')
        self.setRoiAction.triggered.connect(self.right_view.setArenaMode)
        self.right_view.resetArenaAction.triggered.connect(self.resetRoi)
        self.openAction = qw.QAction('Open tracked data (Ctrl+o)')
        self.openAction.triggered.connect(self.openTrackedData)
        self.saveAction = qw.QAction('Save reviewed data (Ctrl+s)')
        self.saveAction.triggered.connect(self.saveReviewedTracks)
        self.speedUpAction = qw.QAction('Double speed (Ctrl+Up arrow)')
        self.speedUpAction.triggered.connect(self.speedUp)
        self.slowDownAction = qw.QAction('Half speed (Ctrl+Down arrow)')
        self.slowDownAction.triggered.connect(self.slowDown)
        self.zoomInLeftAction = qw.QAction('Zoom-in left (+)')
        self.zoomInLeftAction.triggered.connect(self.left_view.zoomIn)
        self.zoomInRightAction = qw.QAction('Zoom-in right (=)')
        self.zoomInRightAction.triggered.connect(self.right_view.zoomIn)
        self.zoomOutLeftAction = qw.QAction('Zoom-out left (Underscore)')
        self.zoomOutLeftAction.triggered.connect(self.left_view.zoomOut)
        self.zoomOutRightAction = qw.QAction('Zoom-out right (-)')
        self.zoomOutRightAction.triggered.connect(self.right_view.zoomOut)
        self.showOldTracksAction = qw.QAction('Show old tracks (o)')
        self.showOldTracksAction.setCheckable(True)
        # self.showOldTracksAction.triggered.connect(self.all_list.setEnabled)
        self.playAction = qw.QAction('Play (Space)')
        self.playAction.triggered.connect(self.playVideo)
        self.resetAction = qw.QAction('Reset')
        self.gotoFrameAction = qw.QAction('Jump to frame (g)')
        self.gotoFrameAction.triggered.connect(self.gotoFrameDialog)
    
        self.frameBreakpointAction = qw.QAction('Set breakpoint at frame (b)')
        self.frameBreakpointAction.triggered.connect(self.setBreakpoint)
        self.curBreakpointAction = qw.QAction('Set breakpoint at current frame (Ctrl+b)')
        self.curBreakpointAction.triggered.connect(self.setBreakpointAtCurrent)
        self.clearBreakpointAction = qw.QAction('Clear frame breakpoint (Shift+b)')
        self.clearBreakpointAction.triggered.connect(self.clearBreakpoint)
        self.jumpToBreakpointAction = qw.QAction('Jump to breakpoint frame (j)')
        self.jumpToBreakpointAction.triggered.connect(self.jumpToBreakpoint)
        self.entryBreakpointAction = qw.QAction('Set breakpoint on appearance (a)')
        self.entryBreakpointAction.triggered.connect(self.setBreakpointAtEntry)
        self.exitBreakpointAction = qw.QAction('Set breakpoint on disappearance (d)')
        self.exitBreakpointAction.triggered.connect(self.setBreakpointAtExit)
        self.clearEntryBreakpointAction = qw.QAction('Clear breakpoint on appearance (Shift+a)')
        self.clearEntryBreakpointAction.triggered.connect(self.clearBreakpointAtEntry)
        self.clearExitBreakpointAction = qw.QAction('Clear breakpoint on disappearance (Shift+d)')
        self.clearExitBreakpointAction.triggered.connect(self.clearBreakpointAtExit)

        self.resetAction.triggered.connect(self.reset)
        self.showDifferenceAction = qw.QAction('Show popup message for left/right mismatch')
        self.showDifferenceAction.setCheckable(True)
        show_difference = settings.value('review/showdiff', 2, type=int)
        self.showDifferenceAction.setChecked(show_difference == 2)
        self.showNewAction = qw.QAction('Show popup message for new tracks')
        self.showNewAction.setCheckable(True)
        self.showNewAction.setChecked(show_difference == 1)
        self.showNoneAction = qw.QAction('No popup message for tracks')
        self.showNoneAction.setCheckable(True)
        self.showNoneAction.setChecked(show_difference == 0)
        self.showHistoryAction = qw.QAction('Show track positions (t)')
        self.showHistoryAction.setCheckable(True)
        self.swapTracksAction = qw.QAction('Swap tracks (drag n drop with right mouse button)')
        self.swapTracksAction.triggered.connect(self.swapTracks)
        self.replaceTrackAction = qw.QAction('Replace track (drag n drop with left mouse button)')
        self.replaceTrackAction.triggered.connect(self.replaceTrack)
        self.deleteTrackAction = qw.QAction('Delete track (Delete/x')
        self.deleteTrackAction.triggered.connect(self.deleteSelected)
        self.undoCurrentChangesAction = qw.QAction('Undo changes in current frame (Ctrl+z)')
        self.undoCurrentChangesAction.triggered.connect(self.undoCurrentChanges)
        self.showLimitsAction = qw.QAction('Size limits')
        self.showLimitsAction.setCheckable(True)
        self.showLimitsAction.setChecked(False)
        self.lim_win.setVisible(False)
        self.showLimitsAction.triggered.connect(self.lim_win.setVisible)
        self.lim_win.sigClose.connect(self.showLimitsAction.setChecked)
        self.histlenAction = qw.QAction('Set oldest tracks to remember')
        self.histlenAction.triggered.connect(self.setHistLen)
        self.histGradientAction = qw.QAction('Set oldest tracks to display')
        self.histGradientAction.triggered.connect(self.setHistGradient)
        self.showChangeListAction = qw.QAction('Show list of changes')
        self.showChangeListAction.triggered.connect(self.showChangeList)
        self.loadChangeListAction = qw.QAction('Load list of changes')
        self.loadChangeListAction.triggered.connect(self.loadChangeList)
        self.saveChangeListAction = qw.QAction('Save list of changes')
        self.saveChangeListAction.triggered.connect(self.saveChangeList)
        self.vidinfoAction = qw.QAction('Video information')
        self.vidinfoAction.triggered.connect(self.vid_info.show)

    def makeShortcuts(self):
        self.sc_play = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Space), self)
        self.sc_play.activated.connect(self.togglePlay)
        self.sc_goto = qw.QShortcut(qg.QKeySequence('G'), self)
        self.sc_goto.activated.connect(self.gotoFrameDialog)
        self.sc_break = qw.QShortcut(qg.QKeySequence('B'), self)
        self.sc_break.activated.connect(self.setBreakpoint)
        self.sc_break_cur = qw.QShortcut(qg.QKeySequence('Ctrl+B'), self)
        self.sc_break_cur.activated.connect(self.setBreakpointAtCurrent)
        self.sc_clear_bp = qw.QShortcut(qg.QKeySequence('Shift+B'), self)
        self.sc_clear_bp.activated.connect(self.clearBreakpoint)
        self.sc_jump_bp = qw.QShortcut(qg.QKeySequence('J'), self)
        self.sc_jump_bp.activated.connect(self.jumpToBreakpoint)
        self.sc_break_appear = qw.QShortcut(qg.QKeySequence('A'), self)
        self.sc_break_appear.activated.connect(self.setBreakpointAtEntry)
        self.sc_break_disappear = qw.QShortcut(qg.QKeySequence('D'), self)
        self.sc_break_disappear.activated.connect(self.setBreakpointAtExit)
        self.sc_clear_appear = qw.QShortcut(qg.QKeySequence('Shift+A'), self)
        self.sc_clear_appear.activated.connect(self.clearBreakpointAtEntry)
        self.sc_clear_disappear = qw.QShortcut(qg.QKeySequence('Shift+D'), self)
        self.sc_clear_disappear.activated.connect(self.clearBreakpointAtExit)
        self.sc_zoom_in_left = qw.QShortcut(qg.QKeySequence('+'), self)
        self.sc_zoom_in_left.activated.connect(self.left_view.zoomIn)
        self.sc_zoom_in_right = qw.QShortcut(qg.QKeySequence('='), self)
        self.sc_zoom_in_right.activated.connect(self.right_view.zoomIn)
        self.sc_zoom_out_right = qw.QShortcut(qg.QKeySequence('-'), self)
        self.sc_zoom_out_right.activated.connect(self.right_view.zoomOut)
        self.sc_zoom_out_left = qw.QShortcut(qg.QKeySequence('_'), self)
        self.sc_zoom_out_left.activated.connect(self.left_view.zoomOut)
        self.sc_old_tracks = qw.QShortcut(qg.QKeySequence('O'), self)
        self.sc_old_tracks.activated.connect(self.showOldTracksAction.toggle)
        self.sc_hist = qw.QShortcut(qg.QKeySequence('T'), self)
        self.sc_hist.activated.connect(self.showHistoryAction.toggle)
        self.sc_next = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_PageDown), self)
        self.sc_next.activated.connect(self.nextFrame)
        self.sc_prev = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_PageUp), self)
        self.sc_prev.activated.connect(self.prevFrame)

        self.sc_remove = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Delete), self)
        self.sc_remove.activated.connect(self.deleteSelected)
        self.sc_remove_2 = qw.QShortcut(qg.QKeySequence('X'), self)
        self.sc_remove_2.activated.connect(self.deleteSelected)
        self.sc_speedup = qw.QShortcut(qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_Up), self)
        self.sc_speedup.activated.connect(self.speedUp)
        self.sc_slowdown = qw.QShortcut(qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_Down), self)
        self.sc_slowdown.activated.connect(self.slowDown)
        self.sc_save = qw.QShortcut(qg.QKeySequence('Ctrl+S'), self)
        self.sc_save.activated.connect(self.saveReviewedTracks)
        self.sc_open = qw.QShortcut(qg.QKeySequence('Ctrl+O'), self)
        self.sc_open.activated.connect(self.openTrackedData)
        self.sc_undo = qw.QShortcut(qg.QKeySequence('Ctrl+Z'), self)
        self.sc_undo.activated.connect(self.undoCurrentChanges)

    @qc.pyqtSlot(bool)
    def disableSeek(self, checked: bool) -> None:
        self.pos_spin.lineEdit().setEnabled(not checked)
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
            if sel not in self.right_tracks:
                continue
            self.right_tracks.pop(sel)
            right_items = self.right_list.findItems(items[0].text(),
                                                    qc.Qt.MatchExactly)
            for item in right_items:
                self.right_list.takeItem(self.right_list.row(item))

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
                self.video_reader is None:  # or \
#                frame_no > self.track_reader.last_frame:
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
    def gotoFrameDialog(self):
        val, ok = qw.QInputDialog.getInt(self, 'Goto frame',
                                         'Jump to frame #',
                                         value=self.frame_no,
                                         min=0)
        if ok and val >= 0:
            self.gotoFrame(val)        

    @qc.pyqtSlot()
    def jumpToBreakpoint(self):
        if self.breakpoint >= 0 and not self.disableSeekAction.isChecked():
            self.gotoFrame(self.breakpoint)

    @qc.pyqtSlot()
    def gotoEditedPos(self):
        frame_no = int(self.pos_spin.text())
        self.gotoFrame(frame_no)

    @qc.pyqtSlot()
    def nextFrame(self):
        self.gotoFrame(self.frame_no + 1)
        if self.play_button.isChecked():
            self.timer.start(int(self.frame_interval / self.speed))

    @qc.pyqtSlot()
    def prevFrame(self):
        if self.frame_no > 0:
            self.gotoFrame(self.frame_no - 1)

    def _flag_tracks(self, all_tracks, cur_tracks):
        """Change the track info to include it age, if age is more than
        `history_length` then remove this track from all tracks -
        this avoids cluttering the view with very old tracks that have not been
        seen in a long time."""
        ret = {}
        for tid, rect in all_tracks.items():
            if self.frame_no - rect[4] <= self.history_length:
               ret[tid] = rect.copy()
        for tid, rect in cur_tracks.items():
            ret[tid] = np.array(rect)
        return ret

    @qc.pyqtSlot()
    def showChangeList(self):
        if self.track_reader is None:
            return
        change_list = [change for change in self.track_reader.change_list
                        if change.frame not in self.track_reader.undone_changes]
        self.changelist_widget.setChangeList(change_list)
        self.changelist_widget.setVisible(True)

    @qc.pyqtSlot()
    def saveChangeList(self):
        if self.track_reader is None:
            return
        fname, _ = qw.QFileDialog.getSaveFileName(self, 'Save list of changes',
                                                  filter='Text (*.csv)')
        if len(fname) > 0:
            self.track_reader.saveChangeList(fname)

    @qc.pyqtSlot()
    def loadChangeList(self):
        if self.track_reader is None:
            return
        fname, _ = qw.QFileDialog.getOpenFileName(self, 'Load list of changes',
                                                  filter='Text (*.csv)')
        if len(fname) > 0:
            self.track_reader.loadChangeList(fname)

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
                vertices = rect2points(np.array(tracks[tid][:4]))  # In reviewer we also pass the frame no. in 5 th element
                contained = [self.roi.containsPoint(qc.QPointF(*vtx), qc.Qt.OddEvenFill)
                             for vtx in vertices]
                if not np.any(contained):
                    self.track_reader.deleteTrack(self.frame_no, tid)
                    tracks.pop(tid)                    
                
        self.old_all_tracks = self.all_tracks.copy()
        self.all_tracks = self._flag_tracks(self.all_tracks, tracks)
        self.sigAllTracksList.emit(list(self.all_tracks.keys()))
        if self.disableSeekAction.isChecked():
            # Going sequentially through frames - copy right to left
            self.frame_no = pos
            self.left_frame = self.right_frame
            self.right_frame = frame
            self.left_tracks = self.right_tracks
            self.right_tracks = self._flag_tracks({}, tracks)
            if self.left_frame is not None:
                self.sigLeftFrame.emit(self.left_frame, pos - 1)
            self.sigRightFrame.emit(self.right_frame, pos)
            if self.showOldTracksAction.isChecked():
                self.sigLeftTracks.emit(self.old_all_tracks)
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
            self._wait_cond.set()
            raise Exception('This should not be reached')
        self.breakpointMessage(pos)
        self.entryExitMessage(self.left_tracks, self.right_tracks)
        message = self._get_diff(self.showNewAction.isChecked())
        if len(message) > 0:
            if self.showDifferenceAction.isChecked() or self.showNewAction.isChecked():
                self.play_button.setChecked(False)
                self.playVideo(False)
                qw.QMessageBox.information(self, 'Track mismatch', message)
            self.sigDiffMessage.emit(message)
        self._wait_cond.set()
        logging.debug('wait condition set')

    def _get_diff(self, show_new):
        right_keys = set(self.right_tracks.keys())
        all_keys = set(self.old_all_tracks.keys())
        new = right_keys - all_keys
        if show_new:
            if len(new) > 0:
                return f'Frame {self.frame_no-1}-{self.frame_no}: New track on right: {new}.'
            return ''
        left_keys = set(self.left_tracks.keys())
        if left_keys != right_keys:
            # logging.info(f'Tracks don\'t match between frames {self.frame_no - 1} '
            #              f'and {self.frame_no}: '
            #              f'{left_keys.symmetric_difference(right_keys)}')
            left_only = left_keys - right_keys
            left_message = f'Tracks only on left: {left_only}.' \
                if len(left_only) > 0 else ''
            right_only = right_keys - left_keys
            right_message = f'Tracks only on right: {right_only}.' \
                if len(right_only) > 0 else ''
            if len(new) > 0:
                right_message += f'New tracks: <b>{new}</b>'
            return f'Frame {self.frame_no - 1}-{self.frame_no}: {left_message} {right_message}'
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

    def setupReading(self, video_path, data_path):
        try:
            self.video_reader = VideoReader(video_path, self._wait_cond)
        except IOError as e:
            qw.QMessageBox.critical(self, 'Error opening video',
                                    f'Could not open video: {video_path}\n'
                                    f'{e}')
            return False
        self.track_reader = TrackReader(data_path)
        self.video_filename = video_path
        self.track_filename = data_path
        settings.setValue('data/directory', os.path.dirname(self.track_filename))
        settings.setValue('video/directory', os.path.dirname(self.video_filename))
        self.vid_info.vidfile.setText(self.video_filename)
        self.breakpoint = self.video_reader.frame_count
        self.vid_info.frames.setText(f'{self.video_reader.frame_count}')
        self.vid_info.fps.setText(f'{self.video_reader.fps}')
        self.vid_info.frame_width.setText(f'{self.video_reader.frame_width}')
        self.vid_info.frame_height.setText(f'{self.video_reader.frame_height}')
        self.left_view.clearAll()
        self.left_view.update()
        self.all_tracks.clear()
        self.left_list.clear()
        self.right_list.clear()
        self.left_tracks = {}
        self.right_tracks = {}
        self.left_frame = None
        self.right_frame = None
        self.history_length = self.track_reader.last_frame
        self.left_view.frame_scene.setHistGradient(self.history_length)
        self.right_view.frame_scene.setHistGradient(self.history_length)
        self.right_view.resetArenaAction.trigger()
        self.lim_widget.sigWmin.connect(self.track_reader.setWmin)
        self.lim_widget.sigWmax.connect(self.track_reader.setWmax)
        self.lim_widget.sigHmin.connect(self.track_reader.setHmin)
        self.lim_widget.sigHmax.connect(self.track_reader.setHmax)
        if self.disableSeekAction.isChecked():
            self.sigNextFrame.connect(self.video_reader.read)
        else:
            self.sigGotoFrame.connect(self.video_reader.gotoFrame)        
        self.video_reader.sigFrameRead.connect(self.setFrame)
        self.video_reader.sigSeekError.connect(self.catchSeekError)
        self.video_reader.sigVideoEnd.connect(self.videoEnd)
        self.frame_interval = 1000.0 / self.video_reader.fps
        self.pos_spin.setRange(0, self.track_reader.last_frame)
        self.slider.setRange(0, self.track_reader.last_frame)
        self.sigChangeTrack.connect(self.track_reader.changeTrack)
        self.track_reader.sigChangeList.connect(self.changelist_widget.setChangeList)
        self.track_reader.sigEnd.connect(self.trackEnd)
        self.sigUndoCurrentChanges.connect(self.track_reader.undoChangeTrack)
        self.sigDataFile.emit(self.track_filename)
        self.gotoFrame(0)
        self.updateGeometry()
        self.tieViews(self.tieViewsAction.isChecked())
        return True

    @qc.pyqtSlot()
    def saveReviewedTracks(self):
        datadir = settings.value('data/directory', '.')
        track_filename, filter = qw.QFileDialog.getSaveFileName(
            self,
            'Save reviewed data',
            datadir, filter='HDF5 (*.h5 *.hdf);; Text (*.csv)')
        logging.debug(f'filename:{track_filename}\nselected filter:{filter}')
        if len(track_filename) > 0:
            if self.save_indicator is None:
                self.save_indicator = qw.QProgressDialog('Saving track data', None,
                                               0,
                                               self.track_reader.last_frame + 1,
                                               self)

                self.save_indicator.setWindowModality(qc.Qt.WindowModal)
                self.save_indicator.resize(400, 200)
                self.track_reader.sigSavedFrames.connect(self.save_indicator.setValue)
            else:                
                self.save_indicator.setRange(0,
                                             self.track_reader.last_frame + 1)
                self.save_indicator.setValue(0)
                try: # make sure same track reader is not connected multiple times
                    self.track_reader.sigSavedFrames.disconnect()
                    self.track_reader.sigSavedFrames.connect(
                        self.save_indicator.setValue)
                except TypeError:
                    pass
            self.save_indicator.show()
            try:
                self.track_reader.saveChanges(track_filename)
                self.save_indicator.setValue(self.track_reader.last_frame + 1)
            except OSError as err:
                qw.QMessageBox.critical(
                    self, 'Error opening file for writing',
                    f'File {track_filename} could not be opened.\n{err}')

    @qc.pyqtSlot()
    def doQuit(self):
        # self._wait_cond.set()
        self.vid_info.close()
        self.changelist_widget.close()

        if self.track_reader is not None and len(self.track_reader.change_list) > 0:
            self.saveReviewedTracks()
        diff = 0
        if self.showNewAction.isChecked():
            diff = 1
        elif self.showDifferenceAction.isChecked():
            diff = 2
        settings.setValue('review/showdiff', diff)
        settings.setValue('review/disable_seek', self.disableSeekAction.isChecked())

    @qc.pyqtSlot(bool)
    def playVideo(self, play: bool):
        if self.video_reader is None:
            return
        if play:
            self.play_button.setText('Pause (Space)')
            self.playAction.setText('Pause (Space)')
            self.timer.start(int(self.frame_interval / self.speed))
        else:
            self.play_button.setText('Play (Space)')
            self.playAction.setText('Play (Space)')
            self.timer.stop()

    @qc.pyqtSlot()
    def togglePlay(self):
        if self.video_reader is None:
            return
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
        self.play_button.setChecked(False)
        self.setupReading(self.video_filename, self.track_filename)

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

    @qc.pyqtSlot()
    def videoEnd(self):
        self.playVideo(False)
        self.play_button.setChecked(False)
        qw.QMessageBox.information(self, 'Finished processing', 'End of video reached.')

    @qc.pyqtSlot()
    def trackEnd(self):
        self.playVideo(False)
        self.play_button.setChecked(False)
        qw.QMessageBox.information(self, 'Finished processing', 'End of tracks reached reached.')
        

class ReviewerMain(qw.QMainWindow):
    sigQuit = qc.pyqtSignal()

    def __init__(self):
        super(ReviewerMain, self).__init__()
        self.review_widget = ReviewWidget()
        file_menu = self.menuBar().addMenu('&File')
        file_menu.addAction(self.review_widget.openAction)
        file_menu.addAction(self.review_widget.saveAction)
        file_menu.addAction(self.review_widget.loadChangeListAction)
        file_menu.addAction(self.review_widget.saveChangeListAction)
        view_menu = self.menuBar().addMenu('&View')
        view_menu.addAction(self.review_widget.zoomInLeftAction)
        view_menu.addAction(self.review_widget.zoomInRightAction)
        view_menu.addAction(self.review_widget.zoomOutLeftAction)
        view_menu.addAction(self.review_widget.zoomOutRightAction)
        view_menu.addAction(self.review_widget.tieViewsAction)
        view_menu.addAction(self.review_widget.autoColorAction)
        view_menu.addAction(self.review_widget.colormapAction)
        view_menu.addAction(self.review_widget.lineWidthAction)
        view_menu.addAction(self.review_widget.showOldTracksAction)
        view_menu.addAction(self.review_widget.showHistoryAction)
        diffgrp = qw.QActionGroup(self)
        diffgrp.addAction(self.review_widget.showDifferenceAction)
        diffgrp.addAction(self.review_widget.showNewAction)
        diffgrp.addAction(self.review_widget.showNoneAction)
        diffgrp.setExclusive(True)
        view_menu.addActions(diffgrp.actions())
        view_menu.addAction(self.review_widget.showLimitsAction)
        view_menu.addAction(self.review_widget.histlenAction)
        view_menu.addAction(self.review_widget.histGradientAction)
        view_menu.addAction(self.review_widget.showChangeListAction)
        view_menu.addAction(self.review_widget.vidinfoAction)

        play_menu = self.menuBar().addMenu('Play')
        play_menu.addAction(self.review_widget.disableSeekAction)
        play_menu.addAction(self.review_widget.playAction)
        play_menu.addAction(self.review_widget.speedUpAction)
        play_menu.addAction(self.review_widget.slowDownAction)
        play_menu.addAction(self.review_widget.resetAction)

        play_menu.addAction(self.review_widget.gotoFrameAction)
        play_menu.addAction(self.review_widget.frameBreakpointAction)
        play_menu.addAction(self.review_widget.curBreakpointAction)
        play_menu.addAction(self.review_widget.entryBreakpointAction)
        play_menu.addAction(self.review_widget.exitBreakpointAction)
        
        play_menu.addAction(self.review_widget.clearBreakpointAction)
        play_menu.addAction(self.review_widget.clearEntryBreakpointAction)
        play_menu.addAction(self.review_widget.clearExitBreakpointAction)

        play_menu.addAction(self.review_widget.jumpToBreakpointAction)

        action_menu = self.menuBar().addMenu('Action')
        action_menu.addActions([self.review_widget.swapTracksAction,
                                self.review_widget.replaceTrackAction,
                                self.review_widget.deleteTrackAction,
                                self.review_widget.undoCurrentChangesAction,
                                self.review_widget.right_view.resetArenaAction])
        self.debugAction = qw.QAction('Debug')
        self.debugAction.setCheckable(True)
        v = settings.value('review/debug', logging.INFO)
        self.setDebug(v == logging.DEBUG)
        self.debugAction.setChecked(v == logging.DEBUG)
        self.debugAction.triggered.connect(self.setDebug)
        action_menu.addAction(self.debugAction)
        toolbar = self.addToolBar('View')
        toolbar.addActions(view_menu.actions())
        toolbar.addActions(action_menu.actions())
        self.review_widget.sigDataFile.connect(self.updateTitle)
        self.sigQuit.connect(self.review_widget.doQuit)
        self.status_label = qw.QLabel()
        self.review_widget.sigDiffMessage.connect(self.status_label.setText)
        self.statusBar().addWidget(self.status_label)
        self.setCentralWidget(self.review_widget)

    @qc.pyqtSlot(bool)
    def setDebug(self, val: bool):
        level = logging.DEBUG if val else logging.INFO
        logging.getLogger().setLevel(level)
        settings.setValue('review/debug', level)

    @qc.pyqtSlot()
    def cleanup(self):
        self.sigQuit.emit()
        settings.sync()
        logging.debug('Saved settings')

    @qc.pyqtSlot(str)
    def updateTitle(self, filename: str) -> None:
        self.setWindowTitle(f'Argos:review {filename}')



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
    zi = qw.QAction('Zoom in (+)')
    zi.triggered.connect(reviewer.left_view.zoomIn)
    zo = qw.QAction('Zoom out (-)')
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
    debug_level = logging.INFO #settings.value('review/debug', logging.INFO)
    logging.getLogger().setLevel(debug_level)
    win = ReviewerMain()
    win.setMinimumSize(800, 600)
    win.setWindowTitle('Argos - review tracks')
    win.showMaximized()
    app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())
