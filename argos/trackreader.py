# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2022-02-20 11:09 PM
import csv
import logging
from datetime import datetime
import time

import numpy as np
import pandas as pd
from PyQt5 import QtCore as qc, QtWidgets as qw, QtGui as qg
from argos.constants import Change, ChangeCode
from sortedcontainers import SortedKeyList
from argos import utility as ut


settings = ut.init()


class TrackReader(qc.QObject):
    """Class to read the tracking data.

    It also keeps a list of changes made by the user and applies them
    before handing over the track information.

    """

    sigEnd = qc.pyqtSignal()
    sigSavedFrames = qc.pyqtSignal(int)
    sigChangeList = qc.pyqtSignal(SortedKeyList)

    def __init__(self, data_file):
        super(TrackReader, self).__init__()
        self.roi = None
        self.data_path = data_file
        if data_file.endswith('.hdf') or data_file.endswith('.h5'):
            self.track_data = pd.read_hdf(self.data_path, 'tracked')
        else:  # assume text file
            has_header = False
            col_count = -1
            # MOT format
            names = [
                'frame',
                'trackid',
                'x',
                'y',
                'w',
                'h',
                'confidence',
                'xc',
                'yc',
                'zc',
            ]
            with open(self.data_path, 'r') as fd:
                reader = csv.reader(fd)
                row = reader.__next__()
                col_count = len(row)
                try:
                    _ = float(row[0])
                    has_header = False
                except ValueError:
                    has_header = True

                if has_header:
                    self.track_data = pd.read_csv(self.data_path)
                else:
                    names = names[:col_count]
                    self.track_data = pd.read_csv(self.data_path, names=names)

        self.track_data = self.track_data.astype(
            {'frame': int, 'trackid': int}
        ).sort_values('frame')
        # Here I keep the entry frames for each track to seatch next
        # new track efficiently
        self.entry_frames = (
            self.track_data.groupby('trackid')
            .frame.aggregate('min')
            .drop_duplicates()
            .sort_values()
        )
        self.track_data.assign(dx=0, dy=0, ds=0)

        for trackid in self.track_data.trackid.unique():
            track = self.track_data[self.track_data.trackid == trackid].copy()
            dx = track['x'].diff().values
            dy = track['y'].diff().values
            ds = np.sqrt(dx**2 + dy**2)
            self.track_data.loc[self.track_data.trackid == trackid, 'dx'] = dx
            self.track_data.loc[self.track_data.trackid == trackid, 'dy'] = dy
            self.track_data.loc[self.track_data.trackid == trackid, 'ds'] = ds
        self.last_frame = self.track_data.frame.max()
        self.wmin = settings.value('segment/min_width', 0)
        self.wmax = settings.value('segment/max_width', 1000)
        self.hmin = settings.value('segment/min_height', 0)
        self.hmax = settings.value('segment/max_height', 1000)

        def keyfn(change):
            return (change.frame, change.idx)

        self.changeList = SortedKeyList(key=keyfn)
        self._change_idx = 0
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

    @qc.pyqtSlot(qg.QPolygonF)
    def setRoi(self, roi):
        self.roi = roi

    @qc.pyqtSlot()
    def resetRoi(self):
        self.roi = None

    def getTrackId(self, track_id, frame_no=None, hist=10):
        """Get all entries for a track across frames.

        Parameters
        ----------
        frame_no: int, default None
            If specified, only entries around this frame are
            returned. If `None`, return all entries for this track.
        hist: int, default 10
            Number of past and future entries around `frame_no`
            to select.

        Returns
        -------
        pd.DataFrame
            The data for track `track_id` for frames `frame_no` -
            `hist` to `frame_no` + `hist`.

            If `frame_no` is `None`, data for `track_id` across all frames.

            `None` if no track matches the specified `track_id` in frame `frame_no`.

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
        return track.iloc[pre:post].copy()

    def getFramePrevNew(self, frame_no):
        """Return the previous frame where a new object ID was detected"""
        fno = -1
        pos = self.entry_frames.searchsorted(frame_no)
        if pos > 0:
            fno = self.entry_frames.iloc[pos - 1]
        return fno

    def getFrameNextNew(self, frame_no):
        """Return the next frame where a new object ID was detected"""
        fno = self.last_frame + 1
        pos = self.entry_frames.searchsorted(frame_no)
        if pos < self.entry_frames.size - 1:
            fno = self.entry_frames.iloc[pos]
            if fno == frame_no:
                fno = self.entry_frames.iloc[pos + 1]
        return fno

    def getFramePrevJump(self, frame_no, threshold):
        """Return the previous frame where any object changed position by over threshold distance"""
        filtered = self.track_data.loc[(self.track_data.frame < frame_no) & (self.track_data.ds > threshold)]
        frame = filtered.frame.max()
        jumped = filtered[(filtered.frame == frame) & (filtered.ds > threshold)]
        return frame, sorted(jumped.trackid.values)

    def getFrameNextJump(self, frame_no, threshold):
        """Return the next frame where any object changed position by over threshold distance"""
        filtered = self.track_data.loc[(self.track_data.frame > frame_no) & (self.track_data.ds > threshold)]
        frame = filtered.frame.min()
        jumped = filtered[(filtered.frame == frame) & (filtered.ds > threshold)]
        for idx, trackid in jumped.trackid.items():
            print(self.track_data[(self.track_data.frame >= frame - 5) & (self.track_data.frame < frame + 5) & (self.track_data.trackid == trackid)])
        return frame, sorted(jumped.trackid.values)

    def getTracks(self, frame_no):
        if frame_no > self.last_frame:
            logging.debug(
                f'Reached last frame with tracks: frame no {frame_no}'
            )
            self.sigEnd.emit()
            return {}
        self.frame_pos = frame_no
        tracks = self.track_data[self.track_data.frame == frame_no]
        # Filter bboxes violating size constraints
        filtered = self._filterTracks(tracks)
        tracks = self.applyChanges(filtered)
        return tracks

    @qc.pyqtSlot(int, int, int)
    def changeTrack(self, frame_no, orig_id, new_id, endFrame=-1):
        """When user assigns `newId` to `orig_id` keep it in undo buffer"""
        change = Change(
            frame=frame_no,
            end=endFrame,
            change=ChangeCode.op_assign,
            orig=orig_id,
            new=new_id,
            idx=self._change_idx,
        )
        self._change_idx += 1
        self.changeList.add(change)
        self.sigChangeList.emit(self.changeList)
        logging.debug(
            f'Changin track: frame: {frame_no}, old: {orig_id}, new: {new_id}'
        )

    @qc.pyqtSlot(int, int, int)
    def swapTrack(self, frameNo, origId, newId, endFrame=-1):
        """When user swaps `newId` with `orig_id` keep it in swap buffer"""
        change = Change(
            frame=frameNo,
            end=endFrame,
            change=ChangeCode.op_swap,
            orig=origId,
            new=newId,
            idx=self._change_idx,
        )
        self._change_idx += 1
        self.changeList.add(change)
        logging.debug(
            f'Swap track: frame: {frameNo}, old: {origId}, new: {newId}'
        )

    def deleteTrack(self, frameNo, origId, endFrame=-1):
        change = Change(
            frame=frameNo,
            end=endFrame,
            change=ChangeCode.op_delete,
            orig=origId,
            new=-1,
            idx=self._change_idx,
        )
        self._change_idx += 1
        self.changeList.add(change)

    @qc.pyqtSlot(int)
    def undoChangeTrack(self, frameNo):
        """This puts the specified frame in a blacklist so all changes applied
        on it are ignored"""
        while True:
            loc = self.changeList.bisect_key_left((frameNo, 0))
            if (
                loc < len(self.changeList)
                and self.changeList[loc].frame == frameNo
            ):
                self.changeList.pop(loc)
            else:
                return

    def _filterTracks(self, tracks):
        wh = np.sort(tracks[['w', 'h']].values, axis=1)
        if self.roi is not None:
            intersects = [
                self.roi.intersects(
                    qg.QPolygonF(qc.QRectF(track.x, track.y, track.w, track.h))
                )
                for track in tracks.itertuples()
            ]
            intersects = np.array(intersects, dtype=bool)
        else:
            intersects = np.ones(tracks.shape[0], dtype=bool)

        sel = np.flatnonzero(
            (wh[:, 0] >= int(self.wmin))
            & (wh[:, 0] <= int(self.wmax))
            & (wh[:, 1] >= int(self.hmin))
            & (wh[:, 1] <= int(self.hmax))
            & intersects
        )
        return tracks.iloc[sel]

    def applyChanges(self, tdata):
        """Apply the changes in `changeList` to traks in `tdata`.

        Parameters
        ----------
        tdata: pd.DataFrame
            Tracks for the current frame

        Returns
        -------
        dict:
            A dict mapping ``trackid: (x, y, w, h, frame)``

        NOTE: `tdata` should have a single `frame` value - changes
        only upto and including this frame are applied.
        """
        if len(tdata) == 0:
            return {}

        tracks = []
        idx_dict = {}
        for ii, row in enumerate(tdata.itertuples()):
            tracks.append([row.trackid, row.x, row.y, row.w, row.h, row.frame])
            idx_dict[row.trackid] = ii
        frameNo = tdata.frame.values[0]
        delete_idx = set()
        for change in self.changeList:
            if change.frame > frameNo:
                break
            if (change.frame in self.undone_changes) or (
                0 <= change.end < frameNo
            ):
                continue
            orig_idx = idx_dict.pop(change.orig, None)
            if change.change == ChangeCode.op_swap:
                new_idx = idx_dict.pop(change.new, None)
                if new_idx is not None:
                    tracks[new_idx][0] = change.orig
                    idx_dict[change.orig] = new_idx
                if orig_idx is not None:
                    tracks[orig_idx][0] = change.new
                    idx_dict[change.new] = orig_idx
            elif (orig_idx is not None) and (
                (change.change == ChangeCode.op_assign)
                or (change.change == ChangeCode.op_merge)
            ):
                # TODO - assign is same as merge now - but maybe in future
                #  differentiate between assign, which should remove
                #  pre-existing change.new item if change.orig is not present
                #  in current tracks, and merge, which should keep
                #  change.new even if change.orig is not present
                tracks[orig_idx][0] = change.new
                new_idx = idx_dict.pop(change.new, None)
                idx_dict[change.new] = orig_idx
                if new_idx is not None:
                    delete_idx.add(new_idx)
            elif (
                change.change == ChangeCode.op_delete
            ) and orig_idx is not None:
                delete_idx.add(orig_idx)
            elif orig_idx is not None:  # push the orig index back
                idx_dict[change.orig] = orig_idx
        tracks = {
            t[0]: t[1:] for ii, t in enumerate(tracks) if ii not in delete_idx
        }
        return tracks

    def saveChanges(self, filepath):
        """Consolidate all the changes made in track id assignment and save.

        Assumptions: as tracking progresses, only new, bigger numbers are
        assigned for track ids. track_id never goes down.

        track ids can be swapped.
        """
        # assignments = self.consolidateChanges()
        data = []
        t1_s = time.perf_counter()
        all_tracks = self._filterTracks(self.track_data)
        for frame_no, fdata in all_tracks.groupby('frame'):
            tracks = self.applyChanges(fdata)
            for tid, tdata in tracks.items():
                data.append([frame_no, tid] + tdata[:4])
                qw.QApplication.processEvents()
            self.sigSavedFrames.emit(frame_no)
        data = pd.DataFrame(
            data=data, columns=['frame', 'trackid', 'x', 'y', 'w', 'h']
        )
        t1_e = time.perf_counter()
        print(
            'Time to apply changes to tracks and combine into data frame',
            t1_e - t1_s,
        )
        t2_s = time.perf_counter()
        changes = [
            (
                change.frame,
                change.end,
                change.change.name,
                change.change.value,
                change.orig,
                change.new,
                change.idx,
            )
            for change in self.changeList
            if change.frame not in self.undone_changes
        ]

        changes = pd.DataFrame(
            data=changes,
            columns=['frame', 'end', 'change', 'code', 'orig', 'new', 'idx'],
        )
        t2_e = time.perf_counter()
        print('Time to collect changes', t2_e - t2_s)
        t3_s = time.perf_counter()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if filepath.endswith('.csv'):
            data.to_csv(filepath, index=False)
            changes.to_csv(f'{filepath}.changelist_{ts}.csv')
        else:
            self._saveDataChangesHDF5(filepath, changes, data, ts)
        t3_e = time.perf_counter()
        print('Time to save data', t3_e - t3_s)
        self.track_data = data
        self.changeList.clear()
        self.sigChangeList.emit(self.changeList)

    def _saveDataChangesHDF5(self, filepath, changes, data, timestamp):
        """Save changes and data in HDF5 format. Here we also store
        the limits (min and max height, width, and roi) as attributes
        of the changes group"""
        if len(changes) == 0:
            return
        with pd.HDFStore(filepath) as store:
            store.put('/tracked', data, format='table')
            change_path = f'changes/changelist_{timestamp}'
            store.put(change_path, changes, format='table')
            store.get_storer(change_path).attrs.wmin = self.wmin
            store.get_storer(change_path).attrs.wmax = self.wmax
            store.get_storer(change_path).attrs.hmin = self.hmin
            store.get_storer(change_path).attrs.hmax = self.hmax
            if self.roi is not None:
                store.get_storer(change_path).attrs.roi = [
                    (int(point.x()), int(point.y())) for point in self.roi
                ]
            else:
                store.get_storer(change_path).attrs.roi = [
                    (0, 0),
                    (0, np.iinfo(np.int32).max),
                    (np.iinfo(np.int32).max, np.iinfo(np.int32).max),
                    (np.iinfo(np.int32).max, 0),
                ]

    @qc.pyqtSlot(str)
    def saveChangeList(self, fname: str) -> None:
        # self.changeList = sorted(self.changeList, key=attrgetter('frame'))
        with open(fname, 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(['frame', 'end', 'change', 'code', 'old', 'new'])
            for change in self.changeList:
                if change.frame not in self.undone_changes:
                    writer.writerow(
                        [
                            change.frame,
                            change.end,
                            change.change.name,
                            change.change.value,
                            change.orig,
                            change.new,
                        ]
                    )

    @qc.pyqtSlot(str)
    def loadChangeList(self, fname: str) -> None:
        self.changeList.clear()
        with open(fname) as fd:
            reader = csv.DictReader(fd)
            idx = 0
            for row in reader:
                new = int(row['new']) if len(row['new']) > 0 else None
                chcode = getattr(ChangeCode, row['change'])
                change = Change(
                    frame=int(row['frame']),
                    end=int(row['end']),
                    change=chcode,
                    orig=int(row['orig']),
                    new=new,
                    idx=idx,
                )
                self.changeList.add(change)
                idx += 1
        self.sigChangeList.emit(self.changeList)
