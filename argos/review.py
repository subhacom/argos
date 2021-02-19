# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-07-09 1:20 PM
"""=========================
Review and correct tracks
=========================
Usage:
::
    python -m argos.review


Basic operation
---------------
At startup it will show a window with two empty panes separated in the
middle by three empty lists titled ``Left tracks``, ``All tracks`` and
``Right tracks`` like :numref:`review_startup` below.

.. _review_startup:
.. figure:: ../doc/images/review_00.png
   :width: 100%
   :alt: Screenshot of review tool at startup

   Screenshot of review tool at startup


To start reviewing tracked data, select ``File->Open tracked data``
from the menubar or press ``Ctrl+O`` on keyboard. This will prompt you
to pick a data file. Once you select the data file, it will then
prompt you to select the corresponding video file. Once done, you
should see the first frame of the video on the right pane with the
bounding boxes (referred to as *bbox* for short) and IDs of the tracked
objects (:numref:`review_loaded`).

.. _review_loaded:
.. figure:: ../doc/images/review_01.png
   :width: 100%
   :alt: Screenshot of review tool after loading data

   Screenshot of review tool after loading data

Here you notice that trackid ``4`` is spurious. So you select it by
clicking on the entry in ``Right tracks`` list. As you select the
enetry, its bbox and ID on the image change color (and line style)
(:numref:`review_select`). If the ``Show track position`` button is
checked, like in the screenshot, then you will also see some points
turning from dark purple to light yellow, indicating all the position
this object takes across the video.

.. _review_select:
.. figure:: ../doc/images/review_02.png
   :width: 100%
   :alt: Screenshot of review tool after selecting object

   Screenshot of review tool after selecting object

Now delete object ``4`` by pressing ``x`` or ``Delete`` on keyboard,
or selecting ``Delete track`` from ``Action`` in menubar
(:numref:`review_delete`).

.. _review_delete:
.. figure:: ../doc/images/review_03.png
   :width: 100%
   :alt: Screenshot of review tool deleting object

   Screenshot of review tool deleting object

Once you delete ``4``, selection will change to the next object
(#``5``) and the path taken by it over time will be displayed in the
same purple-to-yellow color code (:numref:`review_post_delete`).

.. _review_post_delete:
.. figure:: ../doc/images/review_04.png
   :width: 100%
   :alt: Screenshot of review tool after deleting object

   Screenshot of review tool after deleting object, as the next object
   is selected.

Now to play the video, click the ``play`` button at bottom. The right
frame will be transfereed to the left pane, and the next frame will
appear in the right pane.

You will notice the spinbox on bottom right updates the current frame
number as we go forward in the video. Instead of playing the video,
you can also move one frame at a time by clicking the up-arrow in the
spinbox, or by pressing ``PgDn`` on keyboard.

It is useful to pause and inspect the tracks whenever a new object is
dected. In order to pause the video when there is a new trackid, check
the ``Show popup message for new tracks`` item in the ``Diff
settings`` menu (:numref:`review_diff_popup_new`).

.. _review_diff_popup_new:
.. figure:: ../doc/images/review_05.png
   :width: 100%
   :alt: Screenshot Diff settings - popup on new tracks menu

   Enable popup message when a new trackid appears

If you you already played through the video, then all trackids are
old. In order to go back to a prestine state, click the ``Reset``
button at bottom right. If you play the video now, as soon as a new
track appears, the video will pause and a popup message will tell you
the new tracks that appeared between the last frame and the current
frame (:numref:`review_new_track_popup`).

.. _review_new_track_popup:
.. figure:: ../doc/images/review_06.png
   :width: 100%
   :alt: Popup message on new track(s)

   Popup message when a new trackid appears

After you click ``OK`` to dispose of the popup window, the status
message will remind you of the last change
(:numref:`review_status_msg`).

.. _review_status_msg:
.. figure:: ../doc/images/review_07.png
   :width: 100%
   :alt: Status message on new track(s)

   Status message after a new trackid appears

You can also choose ``Show popup message for left/right mismatch`` in
the ``Diff settings`` menu. In this case whenever the trackids on the
left frame are different from those on the right frame, the video will
be paused with a popup message.

If you want to just watch the video without interruption, select ``No
popup message for tracks``.

The other option ``Overlay previous frame``, if selected, will overlay
the previous frame on the right pane in a different color. This may be
helpful for looking at differences between the two frames if the left
and right display is not good enough (:numref:`review_overlay`).

.. _review_overlay:
.. figure:: ../doc/images/review_08.png
   :width: 100%
   :alt: Overlaid previous and current frame.

   Overlaid previous and current frame. The previous frame is in the
   red channel and the current frame in the blue channel, thus
   producing shades of magenta where they have similar values, and
   more red or blue in pixels where they mismatch.


The track lists 
---------------

The three lists between the left and right video frame in the GUI
present the track Ids of the detected objects. These allow you to
display the tracks and carry out modifications of the tracks described
later).

- ``Left tracks`` shows the tracks detected in the left (previous)
  frame. If you select an entry here, its detected track across frames
  will be overlayed on the previous frame in the left pane
  (:numref:`review_track_hist`).

- ``All tracks`` in the middle shows all the tracks seen so far
  (including those that have been lost in the previous or the current
  frame). If you select an entry here, its detected track across
  frames will be overlayed on the previous frame in the left pane. If
  you select different entries in ``Left tracks`` and ``All tracks``,
  the last selected track will be displayed.

- ``Right tracks`` shows the tracks detected in the current frame.  If
  you select an entry here, its detected track across frames will be
  overlayed on the current frame in the right pane.

.. _review_track_hist:
.. figure:: ../doc/images/review_09.png
   :width: 100%
   :alt: Track of the selected object

   The track of the selected object (track Id) in ``Left tracks`` or
   ``All tracks`` is displayed on the left pane. That of the selected
   object in the ``Right tracks`` is displayed on the right pane.


Moving around and break points
------------------------------

To speed up navigation of tracked data, Argos review tool provides
several shortcuts. The corresponding actions are also available in the
``Play`` menu. To play the video, or to stop a video that is already
playing, press the ``Space bar`` on keyboard. You can try to double
the play speed by pressing ``Ctrl + Up Arrow`` and halve the speed by
pressing ``Ctrl + Down Arrow``. The maximum speed is limited by the
time needed to read and display a frame.

Instead of going through the entire video, you can jump to the next
frame where a new trackid was introduced, press ``N`` key (``Jump to
next new track``).

You can jump forward 10 frames by pressing ``Ctrl + PgDn`` and
backward by pressing ``Ctrl + PgUp`` on the keyboard.

To jump to a specific frame number, press ``G`` (``Jump to frame``)
and enter the frame number in the dialog box that pops up.

To remember the current location (frame number) in the video, you can
press ``Ctrl+B`` (``Set breakpoint at current frame``) to set a
breakpoint. You can go to other parts of the video and jump back to
this location by pressing ``J`` (``Jump to breakpoint frame``).  To
clear the breakpoint, press ``Shift+J`` (``Clear frame breakpoint``).

You can set a breakpoint on the appearance of a particular trackid
using ``Set breakpoint on appearance`` (keyboard ``A``), and entering
the track id in the dialog box. When playing the video, it will pause
on the frame where this trackid appears next. Similarly you can set
breakpoint on disappearance of a trackid using ``Set breakpoint on
disappearance`` (keyboard ``D``). You can clear these breakpoints by
pressing ``Shift + A`` and ``Shift + D`` keys respectively.

Finally, if you made any changes (assign, swap, or delete tracks),
then you can jump to the frame corresponding to the next change (after
current frame) by pressing ``C`` and to the last change (before
current frame) by pressing ``Shift + C`` on the keyboard.


Correcting tracks
-----------------
Corrections made in a frame apply to all future frames, unless an operation
is for current-frame only. The past frames are not affected by the changes.
You can undo all changes made in a frame by pressing ``Ctrl+z`` when visiting
that frame.

- Deleting

  You already saw that one can delete spurious tracks by selecting it
  on the ``Right tracks`` list and delete it with ``x`` or ``Delete``
  key.

  To delete a track only in the current frame, but to keep future occurrences
  intact, press ``Ctrl+X`` instead.

- Replacing/Assigning

  Now for example, you can see at frame 111, what has been marked as
  ``12`` was originally animal ``5``, which happened to jump from the
  left wall of the arena to its middle (For this I had to actually
  press ``PgUp`` to go backwards in the video, keeping an eye on this
  animal, until I could be sure where it appeared from). To correct
  the new trackid, we need to assign ``5`` to track id ``12``.

  The easiest way to do this is to use the left mouse button to drag
  the entry ``5`` from either the ``Left tracks`` list or the ``All
  tracks list`` and drop it on entry ``12`` in the ``Right tracks``
  list.  You can also select ``5`` in the left or the middle list and
  ``12`` in the right list and then select ``Replace track`` from the
  ``Action`` menu.

  To apply this only in the current frame keep the ``Shift`` key pressed while
  drag-n-dropping.

- Swapping

  In some cases, especially when one object crosses over another, the
  automatic algorithm can confuse their Ids. You can correct this by
  swapping them.

  To do this, use the right mouse button to drag and drop one entry
  from the ``All tracks`` or ``Left tracks`` list on the other in the
  ``Right tracks`` list. You can also select the track Ids in the
  lists and then click the ``Swap tracks`` entry in the ``Action``
  menu.

  To apply this only in the current frame keep the ``Shift`` key pressed while
  drag-n-dropping.

- Renaming

  To rename a track with a different, nonexistent Id, select the track
  in one of the ``Right tracks`` list and then press the ``R`` key, or
  use the ``Action`` menu to get a prompt for the new Id number. Note
  that normally Argos does not use negative track Id numbers, so for
  temporary use it is safe to use negative numbers and it will not
  conflict with any existing track numbers.

  To apply this only in the current frame keep the ``Shift`` key pressed while
  drag-n-dropping.

All these actions, however, are not immediately made permanent. This
allows you to undo changes that have been made by mistake. You can see
the list of changes you suggested by selecting ``Show list of
changes`` in the view menu, or by using the ``Alt+C`` keyboard
shortcut (:numref:`review_track_changes`). To undo a change, go to the
frame on which it was suggested, and press ``Ctrl+Z``, or select
``Undo changes in current frame`` in the ``Action`` menu.

.. _review_track_changes:
.. figure:: ../doc/images/review_10.png
   :width: 100%
   :alt: List of changes suggested to tracks

   List of changes to be applied to the tracks. The first entry when
   applied will delete the track Id 8 from frame # 24 onwards. The
   last entry will assign the Id 5 to the track 12 in all frames from
   frame # 111 onwards.

You can save the list of changes into a text file with comma separated
values and load them later using entries in the ``File`` menu. The
changes will become permanent once you save the data (``File->Save
reviewed data``). However, the resulting HDF5 file will include the
list of changes in a time-stamped table
:``changes/changelist_YYYYmmdd_HHMMSS``, so you can refer back to past
changes applied to the data

Tips 
---- 
Swapping and assigning on the same trackid within a single frame can
be problematic.  Sometimes the tracking algorithm can temporarily
mislabel tracks. For example, object `A` (ID=1) crosses over
object `B` (ID=2) and after the crossover object `A` got new
label as ID=3, and object `B` got mislabelled as ID=1. The
best order of action here is to 

(a) swap 3 and 1, and then 
(b) assign 2 to 3. 

This is because sometimes the label of `B` gets fixed automatically by
the algorithm after a couple of frames. Since the swap is applied
first, `B`'s 3 becomes 1, but there is no 1 to be switched to 3, thus
there is no trackid 3 in the tracks list, and the assignment does not
happen, and `A` remains 2. Had we first done the assignment and then
the swap, `B` will get the label 2 from the assignment first, and as
`A` also has label 2, both of them will become 1 after the swap.

Sometimes this may not be obvious because the IDs may be lost for a
few frames and later one of the objects re-identified with the old ID
of the other one.

For example this sequence of events may occur: 
1. A(1) approaches B(2).
2. B(2) Id is lost
3. Both A and B get single bounding box with ID 1.
4. A gets new ID 3. B is lost.
5. A has new ID 3, B reappears with 1.

Action sequence to fix this:
1. Go back where A and B have single ID 1.
2. Swap 2 and 1.
3. Go forward when 3 appears on A.
4. Assign 1 to B.

Swapping IDs multiple times can build-up into-hard-to-fix switches
between IDs, as all the changes in the change list buffer are applied
to all future frames. This can be avoided by saving the data
between swaps. This will consolidate all suggested changes in the
buffer and clear the change list.

After swapping two IDs you may notice that one ID keeps jumping between the two
animals. Even if you do the swap again when this happens in later frame, the IDs
keep switching back and forth. In such a case try doing a temporary swap, i.e.,
a swap that applies to the current frame only.

Whenever there are multiple animals getting too close to each other, a
good approach is to put a breakpoint when the algorithm confuses them
for the first time, and slowly go forward several frames to figure out
what the stable IDs become. Also check how long-lived these IDs are
(if a new ID is lost after a few frames, it may be less work to just
delete it, and interpolate the position in between). Then go back and
make the appropriate changes. Remember that the path history uses the
original data read from the track file and does not take into account
any changes you made during a session. To show the updated path, you
have to first save the data so that all your changes are consolidated.

"""

import sys
import os
import csv

from typing import List, Dict
import logging
import threading
from collections import OrderedDict
import numpy as np
from datetime import datetime
import cv2
import pandas as pd
from sortedcontainers import SortedKeyList
from PyQt5 import (
    QtWidgets as qw,
    QtCore as qc,
    QtGui as qg
)

from argos.constants import Change, ChangeCode, change_name, ColorMode
from argos import utility as ut
from argos.utility import make_color, get_cmap_color, rect2points
from argos.frameview import FrameScene, FrameView
from argos.vreader import VideoReader
from argos.limitswidget import LimitsWidget
from argos.vwidget import VidInfo


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
        self.data_path = data_file
        if data_file.endswith('csv'):
            self.track_data = pd.read_csv(self.data_path)
        else:
            self.track_data = pd.read_hdf(self.data_path, 'tracked')
        self.track_data = self.track_data.astype({'frame': int,
                                                  'trackid': int})
        self.last_frame = self.track_data.frame.max()
        self.wmin = 0
        self.wmax = 1000
        self.hmin = 0
        self.hmax = 1000

        def keyfn(change):
            return (change.frame, change.idx)

        self.change_list = SortedKeyList(key=keyfn)
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
        return track.iloc[pre: post].copy()

    def getFrameNextNew(self, frame_no):
        if frame_no > self.last_frame:
            return self.last_frame + 1
        cur_tracks = set(self.track_data[self.track_data.frame <=
                                         frame_no]['trackid'])
        for fno in range(frame_no, self.last_frame + 1):
            tracks = set(self.track_data[self.track_data.frame ==
                                         fno]['trackid'])
            if len(tracks - cur_tracks) > 0:
                return fno
        logging.debug(
            f'Reached last frame with tracks: frame no {self.last_frame}')
        return self.last_frame + 1

    def getTracks(self, frame_no):
        if frame_no > self.last_frame:
            logging.debug(
                f'Reached last frame with tracks: frame no {frame_no}')
            self.sigEnd.emit()
            return {}
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
    def changeTrack(self, frame_no, orig_id, new_id, current=False):
        """When user assigns `new_id` to `orig_id` keep it in undo buffer"""
        change_code = ChangeCode.op_assign_cur if current else ChangeCode.op_assign
        change = Change(frame=frame_no, change=change_code,
                        orig=orig_id, new=new_id,
                        idx=self._change_idx)
        self._change_idx += 1
        self.change_list.add(change)
        self.sigChangeList.emit(self.change_list)
        logging.debug(
            f'Changin track: frame: {frame_no}, old: {orig_id}, new: {new_id}')

    @qc.pyqtSlot(int, int, int)
    def swapTrack(self, frame_no, orig_id, new_id, current=False):
        """When user swaps `new_id` with `orig_id` keep it in swap buffer"""
        change_code = ChangeCode.op_swap_cur if current else ChangeCode.op_swap
        change = Change(frame=frame_no, change=change_code,
                        orig=orig_id, new=new_id,
                        idx=self._change_idx)
        self._change_idx += 1
        self.change_list.add(change)
        logging.debug(
            f'Swap track: frame: {frame_no}, old: {orig_id}, new: {new_id}')

    def deleteTrack(self, frame_no, orig_id, cur=False):
        change_code = ChangeCode.op_delete_cur if cur else ChangeCode.op_delete
        change = Change(frame=frame_no, change=change_code,
                        orig=orig_id, new=None,
                        idx=self._change_idx)
        self._change_idx += 1
        self.change_list.add(change)

    @qc.pyqtSlot(int)
    def undoChangeTrack(self, frame_no):
        """This puts the specified frame in a blacklist so all changes applied
        on it are ignored"""
        while True:
            loc = self.change_list.bisect_key_left((frame_no, 0))
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
        tracks = []
        idx_dict = {}
        for ii, row in enumerate(tdata.itertuples()):
            tracks.append([row.trackid, row.x, row.y, row.w, row.h, row.frame])
            idx_dict[row.trackid] = ii
        frameno = tdata.frame.values[0]
        delete_idx = set()
        for change in self.change_list:
            if change.frame > frameno:
                break
            if change.frame in self.undone_changes:
                continue
            orig_idx = idx_dict.pop(change.orig, None)
            if (change.change == ChangeCode.op_swap) or \
                    (change.change == ChangeCode.op_swap_cur and
                     change.frame == frameno):
                new_idx = idx_dict.pop(change.new, None)
                if new_idx is not None:
                    tracks[new_idx][0] = change.orig
                    idx_dict[change.orig] = new_idx
                if orig_idx is not None:
                    tracks[orig_idx][0] = change.new
                    idx_dict[change.new] = orig_idx
            elif (orig_idx is not None) and \
                    ((change.change == ChangeCode.op_assign) or \
                     (change.change == ChangeCode.op_assign_cur and \
                      change.frame == frameno) or \
                     (change.change == ChangeCode.op_merge)):
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
            elif (change.change == ChangeCode.op_delete or \
                  (change.change == ChangeCode.op_delete_cur and
                   change.frame == frameno)) and orig_idx is not None:
                delete_idx.add(orig_idx)
            elif orig_idx is not None:  # push the orig index back
                idx_dict[change.orig] = orig_idx
        tracks = {t[0]: t[1:] for ii, t in enumerate(tracks) if ii not in delete_idx}
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
            data.to_hdf(filepath, 'tracked', mode='a')
            changes = [(change.frame, change.change.name,
                        change.change.value, change.orig, change.new)
                       for change in self.change_list if change.frame
                       not in self.undone_changes]
            changes = pd.DataFrame(data=changes,
                                   columns=['frame', 'change',
                                            'code', 'orig', 'new'])
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            changes.to_hdf(filepath, f'changes/changelist_{ts}', mode='a')
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
        self.histGradient = 1
        self.trackHist = []
        self.pathCmap = settings.value('review/path_cmap', 'viridis')
        self.markerThickness = settings.value('review/marker_thickness', 2.0)
        self.pathDia = settings.value('review/path_diameter', 5)
        self.trackStyle = '-'  # `-` for line, `o` for ellipse
        # TODO implement choice of the style

    @qc.pyqtSlot(int)
    def setHistGradient(self, age: int) -> None:
        self.histGradient = age

    @qc.pyqtSlot(str)
    def setPathCmap(self, cmap: str) -> None:
        self.pathCmap = cmap
        settings.setValue('review/path_cmap', cmap)

    @qc.pyqtSlot(float)
    def setTrackMarkerThickness(self, thickness: float) -> None:
        """Set the thickness of the marker-edge for drawing paths"""
        self.markerThickness = thickness
        settings.setValue('review/marker_thickness', thickness)

    @qc.pyqtSlot(np.ndarray)
    def showTrackHist(self, track: np.ndarray) -> None:
        for item in self.trackHist:
            self.removeItem(item)
        self.trackHist = []
        if self.trackStyle == '-':
            for ii in range(2, len(track)):
                color = qg.QColor(
                    *get_cmap_color(ii, len(track), self.pathCmap))
                pen = qg.QPen(qg.QBrush(color), self.markerThickness)
                x1, y1 = track[ii - 1]
                x2, y2 = track[ii]
                self.trackHist.append(self.addLine(x1, y1, x2, y2, pen))
        elif self.trackStyle == 'o':
            for ii, t in enumerate(track):
                color = qg.QColor(*get_cmap_color(ii, len(track), self.pathCmap))
                pen = qg.QPen(qg.QBrush(color), self.markerThickness)
                self.trackHist.append(self.addEllipse(t[0],
                                                      t[1],
                                                      self.pathDia,
                                                      self.pathDia,
                                                      pen))
                # self.trackHist = [self.addEllipse(t[0] - 1, t[1] - 1, 2, 2, qg.QPen(self.selected_color)) for t in track]

    @qc.pyqtSlot(float)
    def setPathDia(self, val):
        self.pathDia = val

    @qc.pyqtSlot(dict)
    def setRectangles(self, rects: Dict[int, np.ndarray]) -> None:
        """rects: a dict of id: (x, y, w, h, frame)

        This overrides the same slot in FrameScene where each rectangle has
        a fifth entry indicating frame no of the rectangle.

        The ones from earlier frame that are not present in the current frame
        are displayed with a special line style (default: dashes)
        """
        logging.debug(
            f'{self.objectName()} Received rectangles from {self.sender().objectName()}')
        logging.debug(f'{self.objectName()} Rectangles: {rects}')
        self.clearItems()
        self.trackHist = []
        logging.debug(f'{self.objectName()} cleared')

        for id_, tdata in rects.items():
            if tdata.shape[0] != 5:
                raise ValueError(f'Incorrectly sized entry: {id_}: {tdata}')
            if self.color_mode == ColorMode.auto:
                color = qg.QColor(*make_color(id_))
            elif self.color_mode == ColorMode.cmap:
                color = qg.QColor(
                    *get_cmap_color(id_ % self.max_colors, self.max_colors,
                                    self.colormap))
            else:
                color = qg.QColor(self.color)
            # Use transparency to indicate age
            alpha = 255 * (1 - 0.9 * min(np.abs(self.frameno - tdata[4]),
                                         self.histGradient) / self.histGradient)
            color.setAlpha(int(alpha))
            pen = qg.QPen(color, self.linewidth)
            if tdata[4] != self.frameno:
                pen.setStyle(self.historic_track_ls)
                logging.debug(f'{self.objectName()}: old track : {id_}')
            rect = tdata[:4].copy()
            item = self.addRect(*rect, pen)
            self.itemDict[id_] = item
            text = self.addText(str(id_), self.font)
            self.labelDict[id_] = text
            text.setFont(self.font)
            text.setDefaultTextColor(color)
            # text.setPos(rect[0], rect[1] - text.boundingRect().height())
            text.setPos(rect[0], rect[1])
            text.setFlag(qw.QGraphicsItem.ItemIgnoresTransformations,
                         self.textIgnoresTransformation)
            self.polygons[id_] = rect
            # logging.debug(f'Set {id_}: {rect}')
        if self.arena is not None:
            self.addPolygon(self.arena, qg.QPen(qc.Qt.red))
        self.sigPolygons.emit(self.polygons)
        self.sigPolygonsSet.emit()


class TrackView(FrameView):
    """Visualization of bboxes of objects on video frame with facility to set
    visible area of scene"""
    sigSelected = qc.pyqtSignal(list)
    sigTrackDia = qc.pyqtSignal(float)
    sigTrackMarkerThickness = qc.pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super(TrackView, self).__init__(*args, **kwargs)
        self.sigSelected.connect(self.scene().setSelected)
        self.sigTrackDia.connect(self.scene().setPathDia)
        self.sigTrackMarkerThickness.connect(
            self.scene().setTrackMarkerThickness)

    def setViewportRect(self, rect: qc.QRectF) -> None:
        self.fitInView(rect.x(), rect.y(),
                       rect.width(),
                       rect.height(),
                       qc.Qt.KeepAspectRatio)

    def _makeScene(self):
        self.frameScene = ReviewScene()
        self.setScene(self.frameScene)

    @qc.pyqtSlot()
    def setPathDia(self):
        input_, accept = qw.QInputDialog.getDouble(
            self, 'Diameter of path markers',
            'pixels',
            self.frameScene.pathDia, min=0, max=500)
        if accept:
            self.sigTrackDia.emit(input_)

    @qc.pyqtSlot()
    def setTrackMarkerThickness(self):
        input_, accept = qw.QInputDialog.getDouble(
            self, 'Thickness of path markers',
            'pixels',
            self.frameScene.markerThickness, min=0, max=500)
        if accept:
            self.sigTrackMarkerThickness.emit(input_)


class TrackList(qw.QListWidget):
    sigMapTracks = qc.pyqtSignal(int, int, bool, bool)
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
        """If dragged with left button, assign dropped trackid to the target
        trackid, if right button, swap the two.  If Shift key was
        pressed, then apply these only for the current frame,
        otherwise also all future frames.

        """
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
        current = qw.QApplication.keyboardModifiers() == qc.Qt.ShiftModifier
        self.sigMapTracks.emit(int(source.text()),
                               int(target.text()),
                               self._drag_button == qc.Qt.RightButton,
                               current)
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
        header = self.table.horizontalHeader()
        for ii in range(4):
            # header.setSectionResizeMode(0, qw.QHeaderView.Stretch)
            header.setSectionResizeMode(ii, qw.QHeaderView.ResizeToContents)
        self.setCentralWidget(self.table)

    @qc.pyqtSlot(SortedKeyList)
    def setChangeList(self, change_list):
        self.table.clearContents()
        self.table.setRowCount(len(change_list))
        for ii, change in enumerate(change_list):
            self.table.setItem(ii, 0, qw.QTableWidgetItem(str(change.frame)))
            self.table.setItem(ii, 1, qw.QTableWidgetItem(
                change_name[change.change]))
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
    sigMousePosMessage = qc.pyqtSignal(str)
    sigUndoCurrentChanges = qc.pyqtSignal(int)
    sigDataFile = qc.pyqtSignal(str)
    sigProjectTrackHist = qc.pyqtSignal(np.ndarray)
    sigProjectTrackHistAll = qc.pyqtSignal(np.ndarray)
    sigQuit = qc.pyqtSignal()  # Pass on quit signal in a threadsafe way

    def __init__(self, *args, **kwargs):
        super(ReviewWidget, self).__init__(*args, **kwargs)
        # Keep track of all the tracks seen so far
        self.setObjectName('ReviewWidget')
        self._wait_cond = threading.Event()
        self.breakpoint = -1
        self.entry_break = -1
        self.exit_break = -1
        self.jump_step = 10
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
        self.track_filename = None
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
        self.leftView = TrackView()
        self.leftView.setObjectName('LeftView')
        self.leftView.frameScene.setObjectName('LeftScene')
        # self.leftView.setSizePolicy(qw.QSizePolicy.MinimumExpanding, qw.QSizePolicy.MinimumExpanding)
        self.leftView.setHorizontalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        self.leftView.setVerticalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        panes_layout.addWidget(self.leftView, 1)

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

        self.rightView = TrackView()
        self.rightView.setObjectName('RightView')
        self.rightView.frameScene.setObjectName('RightScene')
        self.rightView.frameScene.setArenaMode()
        # self.rightView.setSizePolicy(qw.QSizePolicy.Expanding,
        #                              qw.QSizePolicy.Expanding)
        self.rightView.setHorizontalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        self.rightView.setVerticalScrollBarPolicy(qc.Qt.ScrollBarAlwaysOn)
        panes_layout.addWidget(self.rightView, 1)
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
        self.sigLeftFrame.connect(self.leftView.setFrame)
        self.sigRightFrame.connect(self.rightView.setFrame)
        self.rightView.sigArena.connect(self.setRoi)
        self.rightView.sigArena.connect(self.leftView.frameScene.setArena)
        self.sigLeftTracks.connect(self.leftView.sigSetRectangles)
        self.sigLeftTrackList.connect(self.left_list.replaceAll)
        self.sigRightTracks.connect(self.rightView.sigSetRectangles)
        self.sigRightTrackList.connect(self.right_list.replaceAll)
        self.sigAllTracksList.connect(self.all_list.replaceAll)
        self.left_list.sigSelected.connect(self.leftView.sigSelected)
        self.all_list.sigSelected.connect(self.projectTrackHist)
        self.all_list.sigSelected.connect(self.leftView.sigSelected)
        self.right_list.sigSelected.connect(self.rightView.sigSelected)
        self.right_list.sigSelected.connect(self.projectTrackHist)
        self.left_list.sigSelected.connect(self.projectTrackHist)
        self.rightView.showBboxAction.triggered.connect(
            self.leftView.showBboxAction.trigger)

        self.rightView.sigTrackDia.connect(
            self.leftView.frameScene.setPathDia)
        self.showBboxAction = self.rightView.showBboxAction
        self.rightView.showIdAction.triggered.connect(
            self.leftView.showIdAction.trigger)
        self.showIdAction = self.rightView.showIdAction
        self.sigProjectTrackHist.connect(
            self.rightView.frameScene.showTrackHist)
        self.sigProjectTrackHistAll.connect(
            self.leftView.frameScene.showTrackHist)
        self.right_list.sigMapTracks.connect(self.mapTracks)
        self.play_button.clicked.connect(self.playVideo)
        self.play_button.setCheckable(True)
        self.slider.valueChanged.connect(self.gotoFrame)
        self.pos_spin.valueChanged.connect(self.gotoFrame)
        self.pos_spin.lineEdit().setEnabled(False)
        self.rightView.sigSetColormap.connect(
            self.leftView.frameScene.setColormap)
        self.rightView.sigSetColor.connect(
            self.leftView.frameScene.setColor)
        # self.sigSetColormap.connect(self.leftView.frameScene.setColormap)
        # self.sigSetColormap.connect(self.rightView.frameScene.setColormap)
        self.leftView.frameScene.sigMousePos.connect(self.mousePosMessage)
        self.rightView.frameScene.sigMousePos.connect(self.mousePosMessage)

    @qc.pyqtSlot(qc.QPointF)
    def mousePosMessage(self, point: qc.QPointF) -> None:
        self.sigMousePosMessage.emit(f'X:{point.x():.02f},Y:{point.y():.02f}')

    @qc.pyqtSlot(list)
    def projectTrackHist(self, selected: list) -> None:
        if not self.showHistoryAction.isChecked():
            return

        for sel in selected:
            if self.sender() == self.right_list:
                track = self.track_reader.getTrackId(sel,
                                                     self.frame_no,
                                                     self.history_length)
            else:
                track = self.track_reader.getTrackId(sel, None)
            if track is None:
                track = np.empty(0)
            else:
                track.loc[:, 'x'] += track.w / 2.0
                track.loc[:, 'y'] += track.h / 2.0
                track = track[['x', 'y']].values
            if self.sender() == self.right_list:
                self.sigProjectTrackHist.emit(track)
            else:
                self.sigProjectTrackHistAll.emit(track)

    @qc.pyqtSlot(Exception)
    def catchSeekError(self, err: Exception) -> None:
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
                                         min=-1000)
        if ok:
            self.entry_break = val

    @qc.pyqtSlot()
    def setBreakpointAtExit(self):
        val, ok = qw.QInputDialog.getInt(self, 'Set breakpoint on exit',
                                         'Pause at disappearance of trackid #',
                                         value=-1,
                                         min=-1000)
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
            value=self.rightView.frameScene.histGradient,
            min=1)
        if ok:
            self.leftView.frameScene.setHistGradient(val)
            self.rightView.frameScene.setHistGradient(val)

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
            logging.debug(f'Before {self.leftView.viewport().size()}')
            logging.debug(f'After {self.rightView.viewport().size()}')
            self.leftView.sigViewportAreaChanged.connect(
                self.rightView.setViewportRect)
            self.rightView.sigViewportAreaChanged.connect(
                self.leftView.setViewportRect)
            self.rightView.horizontalScrollBar().valueChanged.connect(
                self.leftView.horizontalScrollBar().setValue)
            self.rightView.verticalScrollBar().valueChanged.connect(
                self.leftView.verticalScrollBar().setValue)
        else:
            try:
                self.leftView.sigViewportAreaChanged.disconnect()
            except TypeError:
                pass
            try:
                self.rightView.sigViewportAreaChanged.disconnect()
            except TypeError:
                pass
            try:
                self.rightView.horizontalScrollBar().valueChanged.disconnect(
                    self.leftView.horizontalScrollBar().setValue)
            except TypeError:
                pass
            try:
                self.rightView.verticalScrollBar().valueChanged.disconnect(
                    self.leftView.verticalScrollBar().setValue)
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
        self.showGrayscaleAction = self.rightView.showGrayscaleAction
        self.showGrayscaleAction.triggered.connect(
            self.leftView.showGrayscaleAction.trigger)
        self.overlayAction = qw.QAction('Overlay previous frame')
        self.overlayAction.setCheckable(True)
        self.overlayAction.setChecked(False)
        self.invertOverlayColorAction = qw.QAction('Invert overlay color')
        self.invertOverlayColorAction.setCheckable(True)
        self.invertOverlayColorAction.setChecked(False)
        self.setColorAction = self.rightView.setColorAction
        self.autoColorAction = self.rightView.autoColorAction
        self.autoColorAction.triggered.connect(
            self.leftView.autoColorAction.trigger)
        # self.autoColorAction.triggered.connect(self.leftView.autoColorAction)
        self.colormapAction = self.rightView.colormapAction
        self.pathCmapAction = qw.QAction('Set colormap for path')
        self.pathCmapAction.triggered.connect(self.setPathColormap)
        # self.colormapAction = qw.QAction('Colormap')
        # self.colormapAction.triggered.connect(self.setColormap)
        self.lineWidthAction = self.rightView.lineWidthAction
        self.rightView.sigLineWidth.connect(
            self.leftView.frameScene.setLineWidth)
        self.fontSizeAction = self.rightView.fontSizeAction
        self.rightView.sigFontSize.connect(self.leftView.frameScene.setFontSize)
        self.relativeFontSizeAction = self.rightView.relativeFontSizeAction
        self.rightView.frameScene.sigFontSizePixels.connect(
            self.leftView.frameScene.setFontSizePixels)
        self.setPathDiaAction = qw.QAction('Set path marker diameter')
        self.setPathDiaAction.triggered.connect(self.rightView.setPathDia)
        self.setRoiAction = qw.QAction('Set polygon ROI')
        self.setRoiAction.triggered.connect(self.rightView.setArenaMode)
        self.rightView.resetArenaAction.triggered.connect(self.resetRoi)
        self.rightView.resetArenaAction.triggered.connect(
            self.leftView.resetArenaAction.trigger)
        self.openAction = qw.QAction('Open tracked data (Ctrl+o)')
        self.openAction.triggered.connect(self.openTrackedData)
        self.saveAction = qw.QAction('Save reviewed data (Ctrl+s)')
        self.saveAction.triggered.connect(self.saveReviewedTracks)
        self.speedUpAction = qw.QAction('Double speed (Ctrl+Up arrow)')
        self.speedUpAction.triggered.connect(self.speedUp)
        self.slowDownAction = qw.QAction('Half speed (Ctrl+Down arrow)')
        self.slowDownAction.triggered.connect(self.slowDown)
        self.zoomInLeftAction = qw.QAction('Zoom-in left (+)')
        self.zoomInLeftAction.triggered.connect(self.leftView.zoomIn)
        self.zoomInRightAction = qw.QAction('Zoom-in right (=)')
        self.zoomInRightAction.triggered.connect(self.rightView.zoomIn)
        self.zoomOutLeftAction = qw.QAction('Zoom-out left (Underscore)')
        self.zoomOutLeftAction.triggered.connect(self.leftView.zoomOut)
        self.zoomOutRightAction = qw.QAction('Zoom-out right (-)')
        self.zoomOutRightAction.triggered.connect(self.rightView.zoomOut)
        self.showOldTracksAction = qw.QAction('Show old tracks (o)')
        self.showOldTracksAction.setCheckable(True)
        # self.showOldTracksAction.triggered.connect(self.all_list.setEnabled)
        self.playAction = qw.QAction('Play (Space)')
        self.playAction.triggered.connect(self.playVideo)
        self.resetAction = qw.QAction('Reset')
        self.resetAction.setToolTip('Reset to initial state.'
                                    ' Lose all unsaved changes.')
        self.gotoFrameAction = qw.QAction('Jump to frame (g)')
        self.gotoFrameAction.triggered.connect(self.gotoFrameDialog)
        self.jumpForwardAction = qw.QAction('Jump forward (Ctrl+Page down)')
        self.jumpForwardAction.triggered.connect(self.jumpForward)
        self.jumpBackwardAction = qw.QAction('Jump backward (Ctrl+Page up)')
        self.jumpBackwardAction.triggered.connect(self.jumpBackward)
        self.jumpNextNewAction = qw.QAction('Jump to next new track (n)')
        self.jumpNextNewAction.triggered.connect(self.jumpNextNew)
        self.jumpNextChangeAction = qw.QAction('Jump to next change (c)')
        self.jumpNextChangeAction.triggered.connect(self.gotoNextChange)
        self.jumpPrevChangeAction = qw.QAction(
            'Jump to previous change (Shift+c)')
        self.jumpPrevChangeAction.triggered.connect(self.gotoPrevChange)

        self.frameBreakpointAction = qw.QAction('Set breakpoint at frame (b)')
        self.frameBreakpointAction.triggered.connect(self.setBreakpoint)
        self.curBreakpointAction = qw.QAction(
            'Set breakpoint at current frame (Ctrl+b)')
        self.curBreakpointAction.triggered.connect(self.setBreakpointAtCurrent)
        self.clearBreakpointAction = qw.QAction(
            'Clear frame breakpoint (Shift+b)')
        self.clearBreakpointAction.triggered.connect(self.clearBreakpoint)
        self.jumpToBreakpointAction = qw.QAction('Jump to breakpoint frame (j)')
        self.jumpToBreakpointAction.triggered.connect(self.jumpToBreakpoint)
        self.entryBreakpointAction = qw.QAction(
            'Set breakpoint on appearance (a)')
        self.entryBreakpointAction.triggered.connect(self.setBreakpointAtEntry)
        self.exitBreakpointAction = qw.QAction(
            'Set breakpoint on disappearance (d)')
        self.exitBreakpointAction.triggered.connect(self.setBreakpointAtExit)
        self.clearEntryBreakpointAction = qw.QAction(
            'Clear breakpoint on appearance (Shift+a)')
        self.clearEntryBreakpointAction.triggered.connect(
            self.clearBreakpointAtEntry)
        self.clearExitBreakpointAction = qw.QAction(
            'Clear breakpoint on disappearance (Shift+d)')
        self.clearExitBreakpointAction.triggered.connect(
            self.clearBreakpointAtExit)

        self.resetAction.triggered.connect(self.reset)
        self.showDifferenceAction = qw.QAction(
            'Show popup message for left/right mismatch')
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
        self.showHistoryAction.setChecked(True)
        self.swapTracksAction = qw.QAction(
            'Swap tracks (drag n drop with right mouse button)')
        self.swapTracksAction.triggered.connect(self.swapTracks)
        self.swapTracksCurAction = qw.QAction(
            'Swap tracks in current frame only (drag n drop with right mouse '
            'button with Shift-key pressed)')
        self.swapTracksCurAction.triggered.connect(self.swapTracksCur)
        self.replaceTrackAction = qw.QAction(
            'Replace track (drag n drop with left mouse button)')
        self.replaceTrackAction.triggered.connect(self.replaceTrack)
        self.replaceTrackCurAction = qw.QAction(
            'Replace track in current frame only (drag n drop with left mouse '
            'button with Shift-key pressed)')
        self.replaceTrackCurAction.triggered.connect(self.replaceTrackCur)
        self.renameTrackAction = qw.QAction('Rename track (r)')
        self.renameTrackAction.triggered.connect(self.renameTrack)
        self.renameTrackCurAction = qw.QAction(
            'Rename track in current frame (Ctrl+r)')
        self.renameTrackCurAction.triggered.connect(self.renameTrackCur)
        self.deleteTrackAction = qw.QAction('Delete track (Delete/x)')
        self.deleteTrackAction.setToolTip(
            'Keep Ctrl key pressed for only current frame')
        self.deleteTrackAction.triggered.connect(self.deleteSelected)
        self.deleteTrackCurAction = qw.QAction(
            'Delete track only in current frame (Ctrl+Delete/Ctrl+x')
        self.deleteTrackCurAction.triggered.connect(self.deleteSelectedCur)
        self.undoCurrentChangesAction = qw.QAction(
            'Undo changes in current frame (Ctrl+z)')
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
        self.showChangeListAction = qw.QAction('Show list of changes (Alt+c)')
        self.showChangeListAction.setCheckable(True)
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
        # Break point operations
        self.sc_break = qw.QShortcut(qg.QKeySequence('B'), self)
        self.sc_break.activated.connect(self.setBreakpoint)
        self.sc_break_cur = qw.QShortcut(qg.QKeySequence('Ctrl+B'), self)
        self.sc_break_cur.activated.connect(self.setBreakpointAtCurrent)
        self.sc_clear_bp = qw.QShortcut(qg.QKeySequence('Shift+B'), self)
        self.sc_clear_bp.activated.connect(self.clearBreakpoint)
        self.sc_break_appear = qw.QShortcut(qg.QKeySequence('A'), self)
        self.sc_break_appear.activated.connect(self.setBreakpointAtEntry)
        self.sc_break_disappear = qw.QShortcut(qg.QKeySequence('D'), self)
        self.sc_break_disappear.activated.connect(self.setBreakpointAtExit)
        self.sc_clear_appear = qw.QShortcut(qg.QKeySequence('Shift+A'), self)
        self.sc_clear_appear.activated.connect(self.clearBreakpointAtEntry)
        self.sc_clear_disappear = qw.QShortcut(qg.QKeySequence('Shift+D'), self)
        self.sc_clear_disappear.activated.connect(self.clearBreakpointAtExit)
        # Jump in frames
        self.sc_goto = qw.QShortcut(qg.QKeySequence('G'), self)
        self.sc_goto.activated.connect(self.gotoFrameDialog)

        self.sc_jump_bp = qw.QShortcut(qg.QKeySequence('J'), self)
        self.sc_jump_bp.activated.connect(self.jumpToBreakpoint)
        self.sc_jump_fwd = qw.QShortcut(
            qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_PageDown), self)
        self.sc_jump_fwd.activated.connect(self.jumpForward)
        self.sc_jump_back = qw.QShortcut(
            qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_PageUp), self)
        self.sc_jump_back.activated.connect(self.jumpBackward)
        self.sc_jump_nextnew = qw.QShortcut(qg.QKeySequence('N'), self)
        self.sc_jump_nextnew.activated.connect(self.jumpNextNew)
        self.sc_jump_nextchange = qw.QShortcut(qg.QKeySequence('C'), self)
        self.sc_jump_nextchange.activated.connect(self.gotoNextChange)
        self.sc_jump_prevchange = qw.QShortcut(qg.QKeySequence('Shift+C'), self)
        self.sc_jump_prevchange.activated.connect(self.gotoPrevChange)

        self.sc_zoom_in_left = qw.QShortcut(qg.QKeySequence('+'), self)
        self.sc_zoom_in_left.activated.connect(self.leftView.zoomIn)
        self.sc_zoom_in_right = qw.QShortcut(qg.QKeySequence('='), self)
        self.sc_zoom_in_right.activated.connect(self.rightView.zoomIn)
        self.sc_zoom_out_right = qw.QShortcut(qg.QKeySequence('-'), self)
        self.sc_zoom_out_right.activated.connect(self.rightView.zoomOut)
        self.sc_zoom_out_left = qw.QShortcut(qg.QKeySequence('_'), self)
        self.sc_zoom_out_left.activated.connect(self.leftView.zoomOut)
        self.sc_old_tracks = qw.QShortcut(qg.QKeySequence('O'), self)
        self.sc_old_tracks.activated.connect(self.showOldTracksAction.toggle)
        self.sc_hist = qw.QShortcut(qg.QKeySequence('T'), self)
        self.sc_hist.activated.connect(self.showHistoryAction.toggle)
        self.sc_next = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_PageDown), self)
        self.sc_next.activated.connect(self.nextFrame)
        self.sc_prev = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_PageUp), self)
        self.sc_prev.activated.connect(self.prevFrame)

        self.sc_changewin = qw.QShortcut(qg.QKeySequence('Alt+C'), self)
        self.sc_changewin.activated.connect(self.showChangeListAction.trigger)

        self.sc_remove = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Delete), self)
        self.sc_remove.activated.connect(self.deleteSelectedFut)
        self.sc_remove_2 = qw.QShortcut(qg.QKeySequence('X'), self)
        self.sc_remove_2.activated.connect(self.deleteSelectedFut)
        self.sc_remove_cur = qw.QShortcut(
            qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_Delete), self)
        self.sc_remove.activated.connect(self.deleteSelectedCur)
        self.sc_remove_cur_2 = qw.QShortcut(qg.QKeySequence('Ctrl+X'), self)
        self.sc_remove_cur_2.activated.connect(self.deleteSelectedCur)

        self.sc_rename = qw.QShortcut(qg.QKeySequence('R'), self)
        self.sc_rename.activated.connect(self.renameTrack)
        self.sc_rename_cur = qw.QShortcut(qg.QKeySequence('Ctrl+R'), self)
        self.sc_rename_cur.activated.connect(self.renameTrackCur)
        self.sc_speedup = qw.QShortcut(
            qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_Up), self)
        self.sc_speedup.activated.connect(self.speedUp)
        self.sc_slowdown = qw.QShortcut(
            qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_Down), self)
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

    def deleteSelected(self, cur=False) -> None:
        widget = qw.QApplication.focusWidget()
        if isinstance(widget, TrackList):
            items = widget.selectedItems()
        else:
            return
        selected = [int(item.text()) for item in items]
        self.rightView.scene().setSelected(selected)
        self.rightView.scene().removeSelected()
        for sel in selected:
            self.track_reader.deleteTrack(self.frame_no, sel, cur)
            if sel not in self.right_tracks:
                continue
            self.right_tracks.pop(sel)
            right_items = self.right_list.findItems(items[0].text(),
                                                    qc.Qt.MatchExactly)
            for item in right_items:
                self.right_list.takeItem(self.right_list.row(item))

    @qc.pyqtSlot()
    def deleteSelectedFut(self) -> None:
        self.deleteSelected(cur=False)

    @qc.pyqtSlot()
    def deleteSelectedCur(self) -> None:
        self.deleteSelected(cur=True)

    @qc.pyqtSlot()
    def renameTrack(self):
        target = self.right_list.selectedItems()
        if len(target) == 0:
            return
        tid = int(target[0].text())
        val, ok = qw.QInputDialog.getInt(self, 'Rename track',
                                         'New track id:',
                                         value=tid)
        if ok:
            print(f'Renaming track {tid} to {val}')
            self.mapTracks(val, tid, False, False)

    @qc.pyqtSlot()
    def renameTrackCur(self):
        target = self.right_list.selectedItems()
        if len(target) == 0:
            return
        tid = int(target[0].text())
        val, ok = qw.QInputDialog.getInt(self, 'Rename track',
                                         'New track id:',
                                         value=tid)
        if ok:
            print(f'Renaming track {tid} to {val}')
            self.mapTracks(val, tid, False, True)

    @qc.pyqtSlot()
    def swapTracks(self):
        source = self.all_list.selectedItems()
        target = self.right_list.selectedItems()
        if len(source) == 0 or len(target) == 0:
            return
        self.mapTracks(int(source[0].text()), int(target[0].text()),
                       True, False)

    @qc.pyqtSlot()
    def replaceTrack(self):
        source = self.all_list.selectedItems()
        target = self.right_list.selectedItems()
        if len(source) == 0 or len(target) == 0:
            return
        self.mapTracks(int(source[0].text()), int(target[0].text()),
                       False, False)

    @qc.pyqtSlot()
    def swapTracksCur(self):
        source = self.all_list.selectedItems()
        target = self.right_list.selectedItems()
        if len(source) == 0 or len(target) == 0:
            return
        self.mapTracks(int(source[0].text()), int(target[0].text()),
                       True, True)

    @qc.pyqtSlot()
    def replaceTrackCur(self):
        source = self.all_list.selectedItems()
        target = self.right_list.selectedItems()
        if len(source) == 0 or len(target) == 0:
            return
        self.mapTracks(int(source[0].text()), int(target[0].text()),
                       False, True)

    @qc.pyqtSlot(int)
    def gotoFrame(self, frame_no):
        if frame_no < 0:
            frame_no = 0
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
    def jumpForward(self):
        self.gotoFrame(self.frame_no + self.jump_step)

    @qc.pyqtSlot()
    def jumpBackward(self):
        self.gotoFrame(self.frame_no - self.jump_step)

    @qc.pyqtSlot()
    def jumpNextNew(self):
        if self.track_reader is None:
            return
        frame = self.track_reader.getFrameNextNew(self.frame_no)
        self.gotoFrame(frame)

    @qc.pyqtSlot()
    def gotoNextChange(self):
        if self.track_reader is None:
            return
        change_frames = [change.frame for change in
                         self.track_reader.change_list
                         if change.frame not in
                         self.track_reader.undone_changes]
        pos = np.searchsorted(change_frames, self.frame_no, side='right')
        if pos > len(change_frames):
            return
        self.gotoFrame(change_frames[pos])

    @qc.pyqtSlot()
    def gotoPrevChange(self):
        if self.track_reader is None:
            return
        change_frames = [change.frame for change in
                         self.track_reader.change_list
                         if change.frame not in
                         self.track_reader.undone_changes]
        pos = np.searchsorted(change_frames, self.frame_no, side='left') - 1
        if pos < 0:
            return
        self.gotoFrame(change_frames[pos])

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

    @qc.pyqtSlot(bool)
    def showChangeList(self, checked):
        if self.track_reader is None:
            return
        if not checked:
            self.changelist_widget.setVisible(False)
            return
        change_list = [change for change in self.track_reader.change_list
                       if change.frame not in
                       self.track_reader.undone_changes]
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
            if self.left_frame is not None:
                self.sigLeftFrame.emit(self.left_frame, pos - 1)
            self.right_frame = frame
            if self.overlayAction.isChecked() and \
                    (self.left_frame is not None) and len(frame.shape) == 3:
                tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.zeros((frame.shape[0], frame.shape[1], 3),
                                 dtype=np.uint8)
                frame[:, :, 1] = 0
                frame[:, :, 0] = tmp
                frame[:, :, 2] = cv2.cvtColor(self.left_frame,
                                              cv2.COLOR_BGR2GRAY)
                if self.invertOverlayColorAction.isChecked():
                    frame = 255 - frame
            self.sigRightFrame.emit(frame, pos)
            self.left_tracks = self.right_tracks
            self.right_tracks = self._flag_tracks({}, tracks)
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
            self.left_frame = frame
            self.sigLeftFrame.emit(self.left_frame, pos)
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
            self.right_frame = frame
            if self.overlayAction.isChecked() and \
                    (self.left_frame is not None) and len(frame.shape) == 3:
                tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.zeros((frame.shape[0], frame.shape[1], 3),
                                 dtype=np.uint8)
                frame[:, :, 1] = 0
                frame[:, :, 0] = tmp
                frame[:, :, 2] = cv2.cvtColor(self.left_frame,
                                              cv2.COLOR_BGR2GRAY)
                if self.invertOverlayColorAction.isChecked():
                    frame = 255 - frame
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
                return f'Frame {self.frame_no - 1}-{self.frame_no}: New track on right: {new}.'
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
    def setPathColormap(self):
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
                                                 'hot'])
        logging.debug(f'Setting colormap to {input}')
        if not accept:
            return
        self.leftView.frameScene.setPathCmap(input)
        self.rightView.frameScene.setPathCmap(input)

    @qc.pyqtSlot()
    def openTrackedData(self):
        datadir = settings.value('data/directory', '.')
        track_filename, filter = qw.QFileDialog.getOpenFileName(
            self,
            'Open tracked data',
            datadir, filter='HDF5 (*.h5 *.hdf);; Text (*.csv)')
        logging.debug(f'filename:{track_filename}\nselected filter:{filter}')
        if len(track_filename) == 0:
            return
        viddir = os.path.dirname(track_filename)
        vid_filename, vfilter = qw.QFileDialog.getOpenFileName(
            self, 'Open video', viddir)
        logging.debug(f'filename:{vid_filename}\nselected filter:{vfilter}')
        if len(vid_filename) == 0:
            return
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
        settings.setValue('data/directory',
                          os.path.dirname(self.track_filename))
        settings.setValue('video/directory',
                          os.path.dirname(self.video_filename))
        self.vid_info.vidfile.setText(self.video_filename)
        self.breakpoint = self.video_reader.frame_count
        self.vid_info.frames.setText(f'{self.video_reader.frame_count}')
        self.vid_info.fps.setText(f'{self.video_reader.fps}')
        self.vid_info.frame_width.setText(f'{self.video_reader.frame_width}')
        self.vid_info.frame_height.setText(f'{self.video_reader.frame_height}')
        self.leftView.clearAll()
        self.leftView.update()
        self.all_tracks.clear()
        self.left_list.clear()
        self.right_list.clear()
        self.left_tracks = {}
        self.right_tracks = {}
        self.left_frame = None
        self.right_frame = None
        self.history_length = self.track_reader.last_frame
        self.leftView.frameScene.setHistGradient(self.history_length)
        self.rightView.frameScene.setHistGradient(self.history_length)
        self.rightView.resetArenaAction.trigger()
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
        self.sigQuit.connect(self.video_reader.close)
        self.frame_interval = 1000.0 / self.video_reader.fps
        self.pos_spin.blockSignals(True)
        self.pos_spin.setRange(0, self.track_reader.last_frame)
        self.pos_spin.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.track_reader.last_frame)
        self.slider.blockSignals(False)
        self.sigChangeTrack.connect(self.track_reader.changeTrack)
        self.track_reader.sigChangeList.connect(
            self.changelist_widget.setChangeList)
        self.track_reader.sigEnd.connect(self.trackEnd)
        self.sigUndoCurrentChanges.connect(self.track_reader.undoChangeTrack)
        self.sigDataFile.emit(self.track_filename)
        self.gotoFrame(0)
        self.updateGeometry()
        self.tieViews(self.tieViewsAction.isChecked())
        return True

    @qc.pyqtSlot()
    def saveReviewedTracks(self):
        self.playVideo(False)
        datadir = settings.value('data/directory', '.')
        default_file = datadir if self.track_filename is None else self.track_filename
        track_filename, filter = qw.QFileDialog.getSaveFileName(
            self,
            'Save reviewed data',
            default_file,
            filter='HDF5 (*.h5 *.hdf);; Text (*.csv)')
        logging.debug(f'filename:{track_filename}\nselected filter:{filter}')
        if len(track_filename) > 0:
            if self.save_indicator is None:
                self.save_indicator = qw.QProgressDialog('Saving track data',
                                                         None,
                                                         0,
                                                         self.track_reader.last_frame + 1,
                                                         self)

                self.save_indicator.setWindowModality(qc.Qt.WindowModal)
                self.save_indicator.resize(400, 200)
                # self.track_reader.sigSavedFrames.connect(self.save_indicator.setValue)
            else:
                self.save_indicator.setRange(0,
                                             self.track_reader.last_frame + 1)
                self.save_indicator.setValue(0)
            try:  # make sure same track reader is not connected multiple times
                self.track_reader.sigSavedFrames.disconnect()
            except TypeError:
                pass
            self.track_reader.sigSavedFrames.connect(
                self.save_indicator.setValue)
            self.save_indicator.show()
            try:
                self.track_reader.saveChanges(track_filename)
                self.track_reader.data_path = track_filename
                self.track_filename = track_filename
                self.sigDataFile.emit(track_filename)
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
        if self.track_reader is not None and \
                len(self.track_reader.change_list) > 0:
            self.saveReviewedTracks()
        diff = 0
        if self.showNewAction.isChecked():
            diff = 1
        elif self.showDifferenceAction.isChecked():
            diff = 2
        settings.setValue('review/showdiff', diff)
        settings.setValue('review/disable_seek',
                          self.disableSeekAction.isChecked())
        self.sigQuit.emit()

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
        self.leftView.clearAll()
        self.rightView.clearAll()
        self.setupReading(self.video_filename, self.track_filename)

    @qc.pyqtSlot(int, int, bool, bool)
    def mapTracks(self, new_id: int, orig_id: int, swap: bool,
                  cur: bool) -> None:
        if swap:
            self.track_reader.swapTrack(self.frame_no, orig_id, new_id, cur)
        else:
            self.track_reader.changeTrack(self.frame_no, orig_id, new_id, cur)
        tracks = self.track_reader.getTracks(self.frame_no)
        self.sigRightTrackList.emit(list(tracks.keys()))
        self.right_tracks = self._flag_tracks({}, tracks)
        self.sigRightTracks.emit(self.right_tracks)

    @qc.pyqtSlot()
    def videoEnd(self):
        self.playVideo(False)
        self.play_button.setChecked(False)
        qw.QMessageBox.information(self, 'Finished processing',
                                   'End of video reached.')

    @qc.pyqtSlot()
    def trackEnd(self):
        self.playVideo(False)
        self.play_button.setChecked(False)
        qw.QMessageBox.information(self, 'Finished processing',
                                   'End of tracks reached reached.')


class ReviewerMain(qw.QMainWindow):
    sigQuit = qc.pyqtSignal()

    def __init__(self):
        super(ReviewerMain, self).__init__()
        self.reviewWidget = ReviewWidget()
        file_menu = self.menuBar().addMenu('&File')
        file_menu.addAction(self.reviewWidget.openAction)
        file_menu.addAction(self.reviewWidget.saveAction)
        file_menu.addAction(self.reviewWidget.loadChangeListAction)
        file_menu.addAction(self.reviewWidget.saveChangeListAction)

        self.sc_quit = qw.QShortcut(qg.QKeySequence('Ctrl+Q'), self)
        self.sc_quit.activated.connect(self.close)
        self.quitAction = qw.QAction('Quit (Ctrl+Q)')
        self.quitAction.triggered.connect(self.close)
        file_menu.addAction(self.quitAction)

        diff_menu = self.menuBar().addMenu('&Diff settings')
        diff_menu.addAction(self.reviewWidget.overlayAction)
        diff_menu.addAction(self.reviewWidget.invertOverlayColorAction)
        diffgrp = qw.QActionGroup(self)
        diffgrp.addAction(self.reviewWidget.showDifferenceAction)
        diffgrp.addAction(self.reviewWidget.showNewAction)
        diffgrp.addAction(self.reviewWidget.showNoneAction)
        diffgrp.setExclusive(True)
        diff_menu.addActions(diffgrp.actions())

        view_menu = self.menuBar().addMenu('&View')
        view_menu.addAction(self.reviewWidget.tieViewsAction)
        view_menu.addAction(self.reviewWidget.showGrayscaleAction)
        view_menu.addAction(self.reviewWidget.setColorAction)
        view_menu.addAction(self.reviewWidget.autoColorAction)
        view_menu.addAction(self.reviewWidget.colormapAction)
        view_menu.addAction(self.reviewWidget.pathCmapAction)
        view_menu.addAction(self.reviewWidget.fontSizeAction)
        view_menu.addAction(self.reviewWidget.relativeFontSizeAction)
        view_menu.addAction(self.reviewWidget.lineWidthAction)
        view_menu.addAction(self.reviewWidget.setPathDiaAction)
        view_menu.addAction(self.reviewWidget.showBboxAction)
        view_menu.addAction(self.reviewWidget.showIdAction)
        view_menu.addAction(self.reviewWidget.showOldTracksAction)
        view_menu.addAction(self.reviewWidget.showHistoryAction)
        view_menu.addAction(self.reviewWidget.showLimitsAction)
        view_menu.addAction(self.reviewWidget.histlenAction)
        view_menu.addAction(self.reviewWidget.histGradientAction)

        view_menu.addAction(self.reviewWidget.showChangeListAction)
        view_menu.addAction(self.reviewWidget.vidinfoAction)

        zoom_menu = self.menuBar().addMenu('&Zoom')
        zoom_menu.addAction(self.reviewWidget.zoomInLeftAction)
        zoom_menu.addAction(self.reviewWidget.zoomInRightAction)
        zoom_menu.addAction(self.reviewWidget.zoomOutLeftAction)
        zoom_menu.addAction(self.reviewWidget.zoomOutRightAction)

        play_menu = self.menuBar().addMenu('Play')
        play_menu.addAction(self.reviewWidget.disableSeekAction)
        play_menu.addAction(self.reviewWidget.playAction)
        play_menu.addAction(self.reviewWidget.speedUpAction)
        play_menu.addAction(self.reviewWidget.slowDownAction)
        play_menu.addAction(self.reviewWidget.resetAction)

        play_menu.addAction(self.reviewWidget.gotoFrameAction)
        play_menu.addAction(self.reviewWidget.jumpForwardAction)
        play_menu.addAction(self.reviewWidget.jumpBackwardAction)
        play_menu.addAction(self.reviewWidget.frameBreakpointAction)
        play_menu.addAction(self.reviewWidget.curBreakpointAction)
        play_menu.addAction(self.reviewWidget.entryBreakpointAction)
        play_menu.addAction(self.reviewWidget.exitBreakpointAction)

        play_menu.addAction(self.reviewWidget.clearBreakpointAction)
        play_menu.addAction(self.reviewWidget.clearEntryBreakpointAction)
        play_menu.addAction(self.reviewWidget.clearExitBreakpointAction)

        play_menu.addAction(self.reviewWidget.jumpToBreakpointAction)
        play_menu.addAction(self.reviewWidget.jumpNextNewAction)
        play_menu.addAction(self.reviewWidget.jumpNextChangeAction)
        play_menu.addAction(self.reviewWidget.jumpPrevChangeAction)

        action_menu = self.menuBar().addMenu('Action')
        action_menu.addActions([self.reviewWidget.swapTracksAction,
                                self.reviewWidget.replaceTrackAction,
                                self.reviewWidget.replaceTrackCurAction,
                                self.reviewWidget.renameTrackAction,
                                self.reviewWidget.renameTrackCurAction,
                                self.reviewWidget.deleteTrackAction,
                                self.reviewWidget.deleteTrackCurAction,
                                self.reviewWidget.undoCurrentChangesAction,
                                self.reviewWidget.rightView.resetArenaAction])
        self.debugAction = qw.QAction('Debug')
        self.debugAction.setCheckable(True)
        v = settings.value('review/debug', logging.INFO)
        self.setDebug(v == logging.DEBUG)
        self.debugAction.setChecked(v == logging.DEBUG)
        self.debugAction.triggered.connect(self.setDebug)
        action_menu.addAction(self.debugAction)
        toolbar = self.addToolBar('View')
        toolbar.addActions([
            self.reviewWidget.zoomInLeftAction,
            self.reviewWidget.zoomInRightAction,
            self.reviewWidget.zoomOutLeftAction,
            self.reviewWidget.zoomOutRightAction,
            self.reviewWidget.tieViewsAction,
            self.reviewWidget.showOldTracksAction,
            self.reviewWidget.showHistoryAction])

        toolbar.addActions(action_menu.actions())
        self.reviewWidget.sigDataFile.connect(self.updateTitle)
        self.sigQuit.connect(self.reviewWidget.doQuit)
        self.statusLabel = qw.QLabel()
        self.reviewWidget.sigDiffMessage.connect(self.statusLabel.setText)
        self.statusBar().addWidget(self.statusLabel)
        self.posLabel = qw.QLabel()
        self.statusBar().addPermanentWidget(self.posLabel)
        self.reviewWidget.sigMousePosMessage.connect(self.posLabel.setText)
        self.setCentralWidget(self.reviewWidget)

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
    reviewer.leftView.setFrame(image, 0)
    reviewer.rightView.setFrame(image, 0)
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
    zi.triggered.connect(reviewer.leftView.zoomIn)
    zo = qw.QAction('Zoom out (-)')
    zo.triggered.connect(reviewer.rightView.zoomOut)
    arena = qw.QAction('Select arena')
    arena.triggered.connect(reviewer.leftView.scene().setArenaMode)
    arena_reset = qw.QAction('Reset arena')
    arena_reset.triggered.connect(reviewer.rightView.scene().resetArena)
    roi = qw.QAction('Select rectangular ROIs')
    roi.triggered.connect(reviewer.leftView.scene().setRoiRectMode)
    poly = qw.QAction('Select polygon ROIs')
    poly.triggered.connect(reviewer.leftView.scene().setRoiPolygonMode)
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
    debug_level = logging.INFO  # settings.value('review/debug', logging.INFO)
    logging.getLogger().setLevel(debug_level)
    win = ReviewerMain()
    win.setMinimumSize(800, 600)
    win.setWindowTitle('Argos - review tracks')
    win.showMaximized()
    app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())
