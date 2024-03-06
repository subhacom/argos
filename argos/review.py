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
middle by three empty lists titled ``Previous tracks``, ``All tracks`` and
``Current tracks`` like :numref:`review_startup` below.

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
(``# 5``) and the path taken by it over time will be displayed in the
same purple-to-yellow color code (:numref:`review_post_delete`) [#]_.

.. [#] Changing the frame will clear the selection and the path
       display. If you want the selection (and the path-display of the
       selected ID) to be retained across frames, check the menu item
       ``View->Retain selection across frames``.

.. _review_post_delete:
.. figure:: ../doc/images/review_04.png
   :width: 100%
   :alt: Screenshot of review tool after deleting object

   Screenshot of review tool after deleting object, as the next object
   is selected.

Now to play the video, click the ``play`` button at bottom. The right
frame will be transferred to the left pane, and the next frame will
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
previous frame are different from those on the current frame, the video will
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


Selecting a region of interest (ROI)
------------------------------------
If you want to process only a certain part of the frames, you can draw an ROI
by clicking the left mouse-button to set the vertices of a polygon. Click on
the first vertex to close the polygon. If you want to cancel it half-way, click
the right mouse-button.


The track lists
---------------

The three lists between the left (previous) and right (current) video frame in
the GUI present the track Ids of the detected objects. These allow you to
display the tracks and carry out modifications of the tracks described
later).

- ``Previous tracks`` shows the tracks detected in the left (previous)
  frame. If you select an entry here, its detected track across frames
  will be overlayed on the previous frame in the left pane
  (:numref:`review_track_hist`).

- ``All tracks`` in the middle shows all the tracks seen so far
  (including those that have been lost in the previous or the current
  frame). If you select an entry here, its detected track across
  frames will be overlayed on the previous frame in the left pane. If
  you select different entries in ``Left tracks`` and ``All tracks``,
  the last selected track will be displayed.

- ``Current tracks`` shows the tracks detected in the current frame (on the
  right).  If you select an entry here, its detected track across frames will be
  overlayed on the current frame in the right pane.

.. _review_track_hist:
.. figure:: ../doc/images/review_09.png
   :width: 100%
   :alt: Track of the selected object

   The track of the selected object (track Id) in ``Previous tracks`` or
   ``All tracks`` is displayed on the left pane. That of the selected
   object in the ``Current tracks`` is displayed on the right pane.



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

Sometimes tracking algorithms can mix up the identities of two
animals, and this will appear as a sudden jump in the position in the
frame where their ids were swapped. To check these, you can use
``Ctrl+N`` to go to the next jump in position, and ``Ctrl+P`` for the
previous jump. The threshold for the jump can be set in ``Diff
settings`` menu, using the ``Set displacement threshold to flag
jumps``.

You can jump forward 10 frames by pressing ``Ctrl + PgDn`` and
backward by pressing ``Ctrl + PgUp`` on the keyboard.

To jump to a specific frame number, press ``G`` (``Go to frame``)
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
  intact, press ``Shift+X`` instead.

  To apply this from the current frame till a specific frame, press ``Alt+X``.
  A dialog box will appear so you can specify the end frame.

- Replacing/Assigning

  Now for example, you can see at frame 111, what has been marked as
  ``12`` was originally animal ``5``, which happened to jump from the
  left wall of the arena to its middle (For this I had to actually
  press ``PgUp`` to go backwards in the video, keeping an eye on this
  animal, until I could be sure where it appeared from). To correct
  the new trackid, we need to assign ``5`` to track id ``12``.

  The easiest way to do this is to use the left mouse button to drag
  the entry ``5`` from either the ``Previous tracks`` list or the ``All
  tracks list`` and drop it on entry ``12`` in the ``Right tracks``
  list.  You can also select ``5`` in the left or the middle list and
  ``12`` in the right list and then select ``Replace track`` from the
  ``Action`` menu.

  To apply this only in the current frame keep the ``Shift`` key pressed while
  drag-n-dropping.

  To apply this from the current frame till a specific frame, keep the ``Alt``
  key pressed while drag-n-dropping. A dialog box will appear so you can specify
  the end frame.

- Swapping

  In some cases, especially when one object crosses over another, the
  automatic algorithm can confuse their Ids. You can correct this by
  swapping them.

  To do this, use the right mouse button to drag and drop one entry
  from the ``All tracks`` or ``Previous tracks`` list on the other in the
  ``Current tracks`` list. You can also select the track Ids in the
  lists and then click the ``Swap tracks`` entry in the ``Action``
  menu.

  To apply this only in the current frame keep the ``Shift`` key pressed while
  drag-n-dropping.

  To apply this from the current frame till a specific frame, keep the ``Alt``
  key pressed while drag-n-dropping. A dialog box will appear so you can specify
  the end frame.

- Renaming

  To rename a track with a different, nonexistent Id, select the track
  in one of the ``Current tracks`` list and then press the ``R`` key, or
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



Selecting a region of interest
------------------------------

You can exclude some spurious detections by defining a region of interest
in the review tool. Click on the right frame with left-mouse button in order
to start drawing a polygon. Keep clicking to add a vertex at current mouse
cursor position. To close the polygon, click as close to the starting point
as possible. This will crop the frame to polygon and fit it within the view.
As you move forward, any detection outside the polygon will be excluded.


Note on video format
--------------------
Argos capture utility records video in MJPG format in an AVI container.
This is available by default in OpenCV. Although OpenCV can read many
video formats via the ``ffmpeg`` library, most common video formats are
designed for playing sequentially, and jumping back and forth (``seek``)
by arbitrary number of frames is not easy.

With such videos, attempt to jump frames will result in error, and the
review tool will disable ``seek`` when it detects this. To enable seek
when the video format permits it, uncheck the ``Disable seek`` item
in the ``Play`` menu.

"""

import sys
import os
import logging
import threading
from collections import OrderedDict
import numpy as np
import cv2
from sortedcontainers import SortedKeyList
from PyQt5 import QtWidgets as qw, QtCore as qc, QtGui as qg

from argos.constants import (
    change_name,
    STYLE_SHEETS,
)
from argos import utility as ut
from argos.vreader import VideoReader
from argos.limitswidget import LimitsWidget
from argos.vwidget import VidInfo
from argos.tracklist import TrackList
from argos.trackreader import TrackReader
from argos.trackview import TrackView

settings = ut.init()


def displayAbout():
    msgBox = qw.QMessageBox()
    msgBox.setWindowTitle('About Argos Review tool')
    msgBox.setTextFormat(qc.Qt.RichText)
    msgBox.setText(
        """
        Argos tool for reviewing and correcting tracks detected from videos.
        <p>
        Argos is a toolkit for tracking multiple objects from videos. You can get the source-code at <a href='https://github.com/subhacom/argos'>https://github.com/subhacom/argos</a> and the documentation at <a href='https://argos.readthedocs.io/en/latest/'>https://argos.readthedocs.io/en/latest/</a>.
        <p>
        <hr>
        Author: Subhasis Ray
        <br>
        Author-email: lastname dot firstname at gmail dot com
        """,
    )
    msgBox.exec_()


def displayDoc():
    DOC_LINK = 'https://argos.readthedocs.io/en/latest/'
    qg.QDesktopServices.openUrl(qc.QUrl(DOC_LINK))


class LimitWin(qw.QDialog):
    sigClose = qc.pyqtSignal(bool)  # connected to action checked state

    def __init__(self, *args, **kwargs):
        super(LimitWin, self).__init__(*args, **kwargs)
        self.lim_widget = LimitsWidget(self)
        layout = qw.QVBoxLayout()
        layout.addWidget(self.lim_widget)
        self.ok_button = qw.QPushButton('OK', self)
        layout.addWidget(self.ok_button)
        self.setLayout(layout)
        self.ok_button.clicked.connect(self.close)

    def closeEvent(self, a0: qg.QCloseEvent) -> None:
        self.sigClose.emit(False)
        super(LimitWin, self).closeEvent(a0)

    def connectLimitSignals(
        self,
        wminSlot=None,
        wmaxSlot=None,
        hminSlot=None,
        hmaxSlot=None,
        roiSignal=None,
        resetRoiSignal=None,
    ):
        """Connect the signals from `lim_widget` to corresponding
        slots passed as arguments and `roiSignal` to `setRoi` slot"""
        if wminSlot is not None:
            self.lim_widget.sigWmin.connect(wminSlot)
        if wmaxSlot is not None:
            self.lim_widget.sigWmax.connect(wmaxSlot)
        if hminSlot is not None:
            self.lim_widget.sigHmin.connect(hminSlot)
        if hmaxSlot is not None:
            self.lim_widget.sigHmax.connect(hmaxSlot)
        if roiSignal is not None:
            roiSignal.connect(self.lim_widget.setRoi)
        if resetRoiSignal is not None:
            resetRoiSignal.connect(self.lim_widget.resetRoi)


class ChangeWindow(qw.QMainWindow):
    cols = ['frame', 'end', 'change', 'old id', 'new id']

    def __init__(self):
        super(ChangeWindow, self).__init__()
        self.table = qw.QTableWidget()
        self.table.setColumnCount(len(self.cols))
        self.table.setHorizontalHeaderLabels(self.cols)
        self.table.setSizeAdjustPolicy(qw.QAbstractScrollArea.AdjustToContents)
        self.setWindowTitle('Argos: change list')
        header = self.table.horizontalHeader()
        for ii in range(len(self.cols)):
            # header.setSectionResizeMode(0, qw.QHeaderView.Stretch)
            header.setSectionResizeMode(ii, qw.QHeaderView.ResizeToContents)
        self.setCentralWidget(self.table)

    @qc.pyqtSlot(SortedKeyList)
    def setChangeList(self, change_list):
        self.table.clearContents()
        self.table.setRowCount(len(change_list))
        for ii, change in enumerate(change_list):
            self.table.setItem(ii, 0, qw.QTableWidgetItem(str(change.frame)))
            self.table.setItem(ii, 1, qw.QTableWidgetItem(str(change.end)))
            self.table.setItem(
                ii, 2, qw.QTableWidgetItem(change_name[change.change])
            )
            self.table.setItem(ii, 3, qw.QTableWidgetItem(str(change.orig)))
            self.table.setItem(ii, 4, qw.QTableWidgetItem(str(change.new)))
        self.table.resizeColumnsToContents()


class ReviewWidget(qw.QWidget):
    """A widget with two panes for reviewing track mislabelings"""

    sigNextFrame = qc.pyqtSignal()
    sigGotoFrame = qc.pyqtSignal(int)
    sigLeftFrame = qc.pyqtSignal(np.ndarray, int)
    sigRightFrame = qc.pyqtSignal(np.ndarray, int)
    sigLeftTracks = qc.pyqtSignal(dict)
    sigLeftTrackList = qc.pyqtSignal(
        list
    )  # to separate tracks displayed on frame from those in list widget
    sigRightTracks = qc.pyqtSignal(dict)
    sigRightTrackList = qc.pyqtSignal(list)
    sigAllTracksList = qc.pyqtSignal(list)
    sigChangeTrack = qc.pyqtSignal(int, int, int)
    sigSetColormap = qc.pyqtSignal(str, int)
    sigDiffMessage = qc.pyqtSignal(str)
    sigMousePosMessage = qc.pyqtSignal(str)
    sigUndoCurrentChanges = qc.pyqtSignal(int)
    sigDataFile = qc.pyqtSignal(str, bool)
    sigProjectTrackHistLeft = qc.pyqtSignal(np.ndarray, str)
    sigProjectTrackHistRight = qc.pyqtSignal(np.ndarray, str)
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
        self.pathCmap = {
            'left': settings.value('review/path_cmap_left', 'inferno'),
            'right': settings.value('review/path_cmap_right', 'viridis'),
        }
        self.prev = {'frame': None, 'tracks': None, 'pos': -1}
        self.cur = {'frame': None, 'tracks': None, 'pos': -1}
        self.speed = 1.0
        self.timer = qc.QTimer(self)
        self.timer.setSingleShot(True)
        self.video_reader = None
        self.trackReader = None
        self.track_filename = None
        self.vid_info = VidInfo()
        self.roi = None
        # Since video seek is buggy, we have to do continuous reading
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
        label = qw.QLabel('Previous')
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
        list_layout.addWidget(qw.QLabel('Current'))
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
        self.play_button.setIcon(
            self.style().standardIcon(qw.QStyle.SP_MediaPlay)
        )
        self.slider = qw.QSlider(qc.Qt.Horizontal)
        self.pos_spin = qw.QSpinBox()
        self.reset_button = qw.QPushButton('Reset')
        self.reset_button.setIcon(
            self.style().standardIcon(qw.QStyle.SP_MediaStop)
        )
        self.reset_button.clicked.connect(self.reset)
        ctrl_layout = qw.QHBoxLayout()
        ctrl_layout.addWidget(self.play_button)
        ctrl_layout.addWidget(self.slider)
        ctrl_layout.addWidget(self.pos_spin)
        ctrl_layout.addWidget(self.reset_button)
        layout.addLayout(ctrl_layout)
        self.setLayout(layout)
        self.lim_win = LimitWin(self)
        self.changelist_widget = ChangeWindow()
        self.changelist_widget.setVisible(False)
        self.makeActions()
        self.makeShortcuts()
        self.timer.timeout.connect(self.nextFrame)
        self.sigLeftFrame.connect(self.leftView.setFrame)
        self.sigRightFrame.connect(self.rightView.setFrame)
        # self.rightView.sigArena.connect(self.setRoi)
        self.lim_win.connectLimitSignals(
            roiSignal=self.rightView.sigArena,
            resetRoiSignal=self.rightView.resetArenaAction.triggered,
        )
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
            self.leftView.showBboxAction.trigger
        )

        self.rightView.sigTrackDia.connect(self.leftView.frameScene.setPathDia)
        self.showBboxAction = self.rightView.showBboxAction
        self.rightView.showIdAction.triggered.connect(
            self.leftView.showIdAction.trigger
        )
        self.showIdAction = self.rightView.showIdAction
        self.sigProjectTrackHistRight.connect(
            self.rightView.frameScene.showTrackHist
        )
        self.sigProjectTrackHistLeft.connect(
            self.leftView.frameScene.showTrackHist
        )
        self.right_list.sigMapTracks.connect(self.mapTracks)
        self.sigGotoFrame.connect(self.right_list.setCurrentFrame)
        self.play_button.clicked.connect(self.playVideo)
        self.play_button.setCheckable(True)
        self.slider.valueChanged.connect(self.gotoFrame)
        self.pos_spin.valueChanged.connect(self.gotoFrame)
        self.pos_spin.lineEdit().setEnabled(False)
        self.rightView.sigSetColormap.connect(
            self.leftView.frameScene.setColormap
        )
        self.rightView.sigSetColor.connect(self.leftView.frameScene.setColor)
        self.rightView.sigSetSelectedColor.connect(
            self.leftView.frameScene.setSelectedColor
        )
        self.rightView.sigTrackMarkerThickness.connect(
            self.leftView.frameScene.setTrackMarkerThickness
        )
        # self.sigSetColormap.connect(self.leftView.frameScene.setColormap)
        # self.sigSetColormap.connect(self.rightView.frameScene.setColormap)
        self.leftView.frameScene.sigMousePos.connect(self.mousePosMessage)
        self.rightView.frameScene.sigMousePos.connect(self.mousePosMessage)

    @qc.pyqtSlot(qc.QPointF)
    def mousePosMessage(self, point: qc.QPointF) -> None:
        self.sigMousePosMessage.emit(f'X:{point.x():.02f},Y:{point.y():.02f}')

    @qc.pyqtSlot()
    def toggleShowTrackHistory(self) -> None:
        if self.showHistoryAction.isChecked():
            self._projectTrackHist([], 'left')
            self._projectTrackHist([], 'right')
            self.showHistoryAction.setChecked(False)
            return

        if self.left_list.currentItem() is None:
            self._projectTrackHist([], 'left')
        else:
            self._projectTrackHist(
                [int(self.left_list.currentItem().text())], 'left'
            )
        if self.right_list.currentItem() is None:
            self._projectTrackHist([], 'right')
        else:
            self._projectTrackHist(
                [int(self.right_list.currentItem().text())], 'right'
            )

        self.showHistoryAction.setChecked(True)

    def _projectTrackHist(self, selected: list, side: str) -> None:
        """Inner function to send signal to show track on either view.

        Parameters
        ----------
        selected: list
            List containing a single integer (trackid) or an empty list.
        side: str
            Side of the view to display track on, ``left`` or `right``.
        """
        if len(selected) == 0:
            track = None
        else:
            # For both side, bring up full history
            track = self.trackReader.getTrackId(selected[0], None)
        if track is None:
            track = np.empty(0)
        else:
            track.loc[:, 'x'] += track.w / 2.0
            track.loc[:, 'y'] += track.h / 2.0
            track = track[['x', 'y']].values
        # print('Track', track)
        cmap = self.pathCmap[side]
        if side == 'left':
            self.sigProjectTrackHistLeft.emit(track, cmap)
        else:
            self.sigProjectTrackHistRight.emit(track, cmap)

    @qc.pyqtSlot(list)
    def projectTrackHist(self, selected: list) -> None:
        if not self.showHistoryAction.isChecked():
            if len(self.leftView.frameScene.trackHist) > 0:
                self.sigProjectTrackHistLeft.emit(
                    np.empty(0), self.pathCmap['left']
                )
            if len(self.rightView.frameScene.trackHist) > 0:
                self.sigProjectTrackHistRight.emit(
                    np.empty(0), self.pathCmap['right']
                )
            return
        if self.sender() == self.right_list or (
            not self.toggleSideBySideAction.isChecked()
        ):
            self._projectTrackHist(selected, 'right')
        else:
            self._projectTrackHist(selected, 'left')

    @qc.pyqtSlot(Exception)
    def catchSeekError(self, err: Exception) -> None:
        qw.QMessageBox.critical(self, 'Error jumping frames', str(err))
        self.disableSeek(True)
        self.pos_spin.lineEdit().setEnabled(False)
        self.disableSeekAction.setChecked(True)

    @qc.pyqtSlot()
    def setBreakpoint(self):
        val, ok = qw.QInputDialog.getInt(
            self,
            'Set breakpoint',
            'Pause at frame #',
            value=self.breakpoint,
            min=0,
        )
        if ok:
            self.breakpoint = val

    @qc.pyqtSlot()
    def clearBreakpoint(self):
        if self.video_reader is not None:
            self.breakpoint = self.video_reader.frame_count

    @qc.pyqtSlot()
    def setBreakpointAtCurrent(self):
        self.breakpoint = self.cur['pos']

    @qc.pyqtSlot()
    def setBreakpointAtEntry(self):
        val, ok = qw.QInputDialog.getInt(
            self,
            'Set breakpoint at entry',
            'Pause at appearance of trackid #',
            value=-1,
            min=-1000,
        )
        if ok:
            self.entry_break = val

    @qc.pyqtSlot()
    def setBreakpointAtExit(self):
        val, ok = qw.QInputDialog.getInt(
            self,
            'Set breakpoint on exit',
            'Pause at disappearance of trackid #',
            value=-1,
            min=-1000,
        )
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
                self,
                'Processing paused',
                f'Reached breakpoint at frame # {self.breakpoint}',
            )

    def entryExitMessage(self, left, right):
        do_break = False
        if (
            (self.entry_break in right)
            and (left is not None)
            and (self.entry_break not in left)
        ):
            do_break = True
            message = f'Reached breakpoint at entry of # {self.entry_break}'
        elif (
            (left is not None)
            and (self.exit_break in left)
            and (self.exit_break not in right)
        ):
            do_break = True
            message = f'Reached breakpoint at exit of # {self.exit_break}'
        if do_break:
            self.play_button.setChecked(False)
            self.playVideo(False)
            qw.QMessageBox.information(self, 'Processing paused', message)

    @qc.pyqtSlot()
    def setHistLen(self):
        val, ok = qw.QInputDialog.getInt(
            self,
            'History length',
            'Oldest tracks to keep (# of frames)',
            value=self.history_length,
            min=1,
        )
        if ok:
            self.history_length = val

    @qc.pyqtSlot()
    def setHistGradient(self):
        val, ok = qw.QInputDialog.getInt(
            self,
            'Color-gradient for  old tracks',
            'Faintest of old tracks (# of frames)',
            value=self.rightView.frameScene.histGradient,
            min=1,
        )
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
                self.rightView.setViewportRect
            )
            self.rightView.sigViewportAreaChanged.connect(
                self.leftView.setViewportRect
            )
            self.rightView.horizontalScrollBar().valueChanged.connect(
                self.leftView.horizontalScrollBar().setValue
            )
            self.rightView.verticalScrollBar().valueChanged.connect(
                self.leftView.verticalScrollBar().setValue
            )
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
                    self.leftView.horizontalScrollBar().setValue
                )
            except TypeError:
                pass
            try:
                self.rightView.verticalScrollBar().valueChanged.disconnect(
                    self.leftView.verticalScrollBar().setValue
                )
            except TypeError:
                pass

    def wheelEvent(self, event):
        steps = event.angleDelta().y() // 120
        # print('Mouse wheel steps', steps)
        direction = steps and steps // abs(steps)  # 0, 1, or -1
        # print(event.modifiers())
        if event.modifiers() == qc.Qt.ShiftModifier:
            fwd = self.nextFrame
            bak = self.prevFrame
        elif (event.modifiers() & qc.Qt.ControlModifier) and (
            event.modifiers() & qc.Qt.ShiftModifier
        ):
            fwd = self.jumpForward
            bak = self.jumpBackward
        else:
            super(ReviewWidget, self).wheelEvent(event)
            return

        for step in range(1, abs(steps) + 1):
            if direction == -1:
                fwd()
            elif direction == 1:
                bak()
            if self.cur['pos'] == self.breakpoint:
                break

    def makeActions(self):
        self.disableSeekAction = qw.QAction('Disable seek')
        self.disableSeekAction.setToolTip(
            'Most video formats do not allow '
            'jumping back and forth by arbitrary number of frames.'
            ' Checking this prevents errors when processing such videos.'
        )
        self.disableSeekAction.setCheckable(True)
        self.disableSeek(False)
        self.disableSeekAction.setChecked(False)
        self.disableSeekAction.triggered.connect(self.disableSeek)
        self.tieViewsAction = qw.QAction('Scroll views together')
        self.tieViewsAction.setCheckable(True)
        self.tieViewsAction.triggered.connect(self.tieViews)
        self.tieViewsAction.setChecked(True)
        self.tieViews(True)
        self.showGrayscaleAction = self.rightView.showGrayscaleAction
        self.showGrayscaleAction.triggered.connect(
            self.leftView.showGrayscaleAction.trigger
        )
        self.overlayAction = qw.QAction('Overlay previous frame')
        self.overlayAction.setCheckable(True)
        self.overlayAction.setChecked(False)
        self.invertOverlayColorAction = qw.QAction('Invert overlay color')
        self.invertOverlayColorAction.setCheckable(True)
        self.invertOverlayColorAction.setChecked(False)
        self.keepSelectionAction = qw.QAction('Retain selection across frames')
        self.keepSelectionAction.setToolTip(
            'If unchecked, item selection is '
            'lost when frame changes. If '
            'checked and path display is on,'
            'this can make things very slow.'
        )
        self.keepSelectionAction.setCheckable(True)
        self.keepSelectionAction.setChecked(self.right_list.keepSelection)
        self.keepSelectionAction.triggered.connect(
            self.right_list.setKeepSelection
        )
        self.keepSelectionAction.triggered.connect(
            self.left_list.setKeepSelection
        )
        self.keepSelectionAction.triggered.connect(
            self.all_list.setKeepSelection
        )
        self.setColorAction = self.rightView.setColorAction
        self.setSelectedColorAction = self.rightView.setSelectedColorAction
        self.setAlphaUnselectedAction = self.rightView.setAlphaUnselectedAction
        self.rightView.sigSetAlphaUnselected.connect(
            self.leftView.frameScene.setAlphaUnselected
        )
        self.autoColorAction = self.rightView.autoColorAction
        self.autoColorAction.triggered.connect(
            self.leftView.autoColorAction.trigger
        )
        # self.autoColorAction.triggered.connect(self.leftView.autoColorAction)
        self.colormapAction = self.rightView.colormapAction
        self.pathCmapLeftAction = qw.QAction(
            'Colormap for path in previous frame'
        )
        self.pathCmapLeftAction.triggered.connect(self.setPathCmapLeft)
        self.pathCmapRightAction = qw.QAction(
            'Colormap for path in current frame'
        )
        self.pathCmapRightAction.triggered.connect(self.setPathCmapRight)
        # self.colormapAction = qw.QAction('Colormap')
        # self.colormapAction.triggered.connect(self.setColormap)
        self.labelInsideAction = self.rightView.setLabelInsideAction
        self.labelInsideAction.triggered.connect(
            self.leftView.frameScene.setLabelInside
        )
        self.lineWidthAction = self.rightView.lineWidthAction
        self.rightView.sigLineWidth.connect(
            self.leftView.frameScene.setLineWidth
        )
        self.fontSizeAction = self.rightView.fontSizeAction
        self.rightView.sigFontSize.connect(
            self.leftView.frameScene.setFontSize
        )
        self.relativeFontSizeAction = self.rightView.relativeFontSizeAction
        self.rightView.frameScene.sigFontSizePixels.connect(
            self.leftView.frameScene.setFontSizePixels
        )
        self.setPathDiaAction = qw.QAction('Set path marker diameter')
        self.setPathDiaAction.triggered.connect(self.rightView.setPathDia)
        self.setMarkerThicknessAction = qw.QAction('Set path linewidth')
        self.setMarkerThicknessAction.triggered.connect(
            self.rightView.setTrackMarkerThickness
        )
        self.setRoiAction = qw.QAction('Set polygon ROI')
        self.setRoiAction.triggered.connect(self.rightView.setArenaMode)
        self.enableDrawArenaAction = qw.QAction('Draw arena')
        self.enableDrawArenaAction.setCheckable(True)
        self.enableDrawArenaAction.triggered.connect(self.rightView.enableDraw)
        self.enableDrawArenaAction.triggered.connect(self.leftView.enableDraw)
        self.enableDrawArenaAction.setChecked(False)
        # self.rightView.resetArenaAction.triggered.connect(self.resetRoi)

        self.rightView.resetArenaAction.triggered.connect(
            self.leftView.resetArenaAction.trigger
        )
        self.openAction = qw.QAction('Open tracked data (Ctrl+o)')
        self.openAction.setIcon(
            self.style().standardIcon(qw.QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        self.openAction.triggered.connect(self.openTrackedData)
        self.saveAction = qw.QAction('Save reviewed data (Ctrl+s)')
        self.saveAction.setIcon(
            self.style().standardIcon(
                qw.QStyle.StandardPixmap.SP_DialogSaveButton
            )
        )
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
        self.playAction.setIcon(
            self.style().standardIcon(qw.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.playAction.triggered.connect(self.playVideo)
        self.resetAction = qw.QAction('Reset')
        self.resetAction.setIcon(
            self.style().standardIcon(qw.QStyle.StandardPixmap.SP_MediaStop)
        )
        self.resetAction.setToolTip(
            'Reset to initial state.' ' Lose all unsaved changes.'
        )

        self.nextFrameAction = qw.QAction('Next frame (Page down)')
        self.nextFrameAction.triggered.connect(self.nextFrame)
        self.prevFrameAction = qw.QAction('Previous frame (Page up)')
        self.prevFrameAction.triggered.connect(self.prevFrame)
        self.gotoFrameAction = qw.QAction('Go to frame (g)')
        self.gotoFrameAction.triggered.connect(self.gotoFrameDialog)
        self.jumpForwardAction = qw.QAction('Jump forward (Ctrl+Page down)')
        self.jumpForwardAction.triggered.connect(self.jumpForward)
        self.jumpBackwardAction = qw.QAction('Jump backward (Ctrl+Page up)')
        self.jumpBackwardAction.triggered.connect(self.jumpBackward)

        self.jumpNextNewAction = qw.QAction('Go to next new track (n)')
        self.jumpNextNewAction.triggered.connect(self.jumpNextNew)
        self.jumpPrevNewAction = qw.QAction('Go to previous new track (p)')
        self.jumpPrevNewAction.triggered.connect(self.jumpPrevNew)

        self.jumpNextChangeAction = qw.QAction('Go to next change (c)')
        self.jumpNextChangeAction.triggered.connect(self.jumpNextChange)
        self.jumpPrevChangeAction = qw.QAction(
            'Go to previous change (Shift+c)'
        )
        self.jumpPrevChangeAction.triggered.connect(self.jumpPrevChange)
        self.jumpNextJumpAction = qw.QAction('Go to next jump (Ctrl+n)')
        self.jumpNextJumpAction.triggered.connect(self.jumpNextJump)
        self.jumpPrevJumpAction = qw.QAction('Go to previous jump (Ctrl+p)')
        self.jumpPrevJumpAction.triggered.connect(self.jumpPrevJump)

        self.frameBreakpointAction = qw.QAction('Set breakpoint at frame (b)')
        self.frameBreakpointAction.triggered.connect(self.setBreakpoint)
        self.curBreakpointAction = qw.QAction(
            'Set breakpoint at current frame (Ctrl+b)'
        )
        self.curBreakpointAction.triggered.connect(self.setBreakpointAtCurrent)
        self.clearBreakpointAction = qw.QAction(
            'Clear frame breakpoint (Shift+b)'
        )
        self.clearBreakpointAction.triggered.connect(self.clearBreakpoint)
        self.jumpToBreakpointAction = qw.QAction(
            'Jump to breakpoint frame (j)'
        )
        self.jumpToBreakpointAction.triggered.connect(self.jumpToBreakpoint)
        self.entryBreakpointAction = qw.QAction(
            'Set breakpoint on appearance (a)'
        )
        self.entryBreakpointAction.triggered.connect(self.setBreakpointAtEntry)
        self.exitBreakpointAction = qw.QAction(
            'Set breakpoint on disappearance (d)'
        )
        self.exitBreakpointAction.triggered.connect(self.setBreakpointAtExit)
        self.clearEntryBreakpointAction = qw.QAction(
            'Clear breakpoint on appearance (Shift+a)'
        )
        self.clearEntryBreakpointAction.triggered.connect(
            self.clearBreakpointAtEntry
        )
        self.clearExitBreakpointAction = qw.QAction(
            'Clear breakpoint on disappearance (Shift+d)'
        )
        self.clearExitBreakpointAction.triggered.connect(
            self.clearBreakpointAtExit
        )

        self.resetAction.triggered.connect(self.reset)
        self.showDifferenceAction = qw.QAction(
            'Show popup message for left/right mismatch'
        )
        self.showDifferenceAction.setCheckable(True)
        show_difference = settings.value('review/showdiff', 2, type=int)
        self.showDifferenceAction.setChecked(show_difference == 2)
        self.setJumpThresholdAction = qw.QAction('Set displacement threshold to flag jumps')
        self.setJumpThresholdAction.triggered.connect(self._setJumpThreshold)
        self.showNewAction = qw.QAction('Show popup message for new tracks')
        self.showNewAction.setCheckable(True)
        self.showNewAction.setChecked(show_difference == 1)
        self.showNoneAction = qw.QAction('No popup message for tracks')
        self.showNoneAction.setCheckable(True)
        self.showNoneAction.setChecked(show_difference == 0)
        self.showHistoryAction = qw.QAction('Show track positions (t)')
        self.showHistoryAction.setCheckable(True)
        self.showHistoryAction.setChecked(True)
        self.showHistoryAction.triggered.connect(self.toggleShowTrackHistory)
        self.swapTracksAction = qw.QAction('Swap tracks')
        self.swapTracksAction.setToolTip('Drag n drop with right mouse button')
        self.swapTracksAction.triggered.connect(self.swapTracks)
        self.swapTracksCurAction = qw.QAction(
            'Swap tracks in current frame only'
        )
        self.swapTracksAction.setToolTip(
            'Keep Shift-key pressed and using right mouse button drag track ID'
            ' from Previous/All list and drop over target ID in right list'
        )
        self.swapTracksCurAction.triggered.connect(self.swapTracksCur)
        self.swapTracksRangeAction = qw.QAction(
            'Swap tracks up to a specific frame'
        )
        self.swapTracksRangeAction.setToolTip(
            'Drag and drop with right mouse button with Alt-key pressed'
        )
        self.swapTracksRangeAction.triggered.connect(self.swapTracksRange)
        self.replaceTrackAction = qw.QAction('Replace track')
        self.replaceTrackAction.setToolTip(
            'Using left mouse button drag track ID from left or middle list and'
            ' drop on target track ID in right pane'
        )
        self.replaceTrackAction.triggered.connect(self.replaceTrack)
        self.replaceTrackCurAction = qw.QAction('Replace in current frame')
        self.replaceTrackCurAction.setToolTip(
            'Keep Shift-key pressed and using left mouse button drag track ID'
            ' from left/middle list and drop over target track ID in right pane'
        )
        self.replaceTrackCurAction.triggered.connect(self.replaceTrackCur)
        self.replaceTracksRangeAction = qw.QAction(
            'Replace track up to a specific frame'
        )
        self.replaceTrackAction.setToolTip(
            'Keep Alt-key and using left mouse button to drag ID from'
            ' left/middle list and drop over targte ID in right list'
        )
        self.swapTracksRangeAction.triggered.connect(self.replaceTracksRange)
        self.renameTrackAction = qw.QAction('Rename track (r)')
        self.renameTrackAction.setToolTip(
            'Rename selected track (shortcut: r key)'
        )
        self.renameTrackAction.triggered.connect(self.renameTrack)
        self.renameTrackCurAction = qw.QAction('Rename track in current frame')
        self.renameTrackCurAction.setToolTip(
            'Rename selected track in current frame only (shortcut: Shift+r key)'
        )
        self.renameTrackCurAction.triggered.connect(self.renameTrackCur)
        self.deleteTrackAction = qw.QAction('Delete track (x or Del)')
        self.deleteTrackAction.setToolTip(
            'Delete selected track (shortcut: Delete or x key)'
        )
        self.deleteTrackAction.triggered.connect(self.deleteSelected)
        self.deleteTrackCurAction = qw.QAction('Delete only in current frame')
        self.deleteTrackCurAction.setToolTip(
            'Shortcut: Shift+Delete or Shift+x key'
        )
        self.deleteTrackCurAction.triggered.connect(self.deleteSelectedCur)
        self.deleteTrackRangeAction = qw.QAction(
            'Delete track up to a specific frame'
        )
        self.deleteTrackRangeAction.setToolTip('Shortcut: Alt+Delete or Alt+x')
        self.deleteTrackRangeAction.triggered.connect(self.deleteSelectedRange)
        self.undoCurrentChangesAction = qw.QAction(
            'Undo changes in current frame'
        )
        self.undoCurrentChangesAction.setToolTip('Shortcut: Ctrl+z')
        self.undoCurrentChangesAction.triggered.connect(
            self.undoCurrentChanges
        )
        self.showLimitsAction = qw.QAction('Set size limits')
        self.showLimitsAction.setToolTip(
            'Set size limits on bboxes of tracks to be displayed'
        )
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
        self.showChangeListAction.triggered.connect(self.updateChangeList)
        self.sigDataFile.connect(self.updateChangeList)
        self.loadChangeListAction = qw.QAction('Load list of changes')
        self.loadChangeListAction.triggered.connect(self.loadChangeList)
        self.saveChangeListAction = qw.QAction('Save list of changes')
        self.saveChangeListAction.triggered.connect(self.saveChangeList)
        self.toggleSideBySideAction = qw.QAction('View side-by-side')
        self.toggleSideBySideAction.setToolTip(
            'Show previous and current frame side by side'
        )
        self.toggleSideBySideAction.setCheckable(True)
        checked = settings.value('review/sidebyside', True, type=bool)
        self.toggleSideBySideAction.setChecked(checked)
        self.toggleSideBySideAction.triggered.connect(
            self.toggleSideBySideView
        )
        self.toggleSideBySideView(checked)
        self.vidinfoAction = qw.QAction('Video information')
        self.vidinfoAction.setIcon(
            self.style().standardIcon(
                qw.QStyle.StandardPixmap.SP_FileDialogInfoView
            )
        )
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
        self.sc_clear_disappear = qw.QShortcut(
            qg.QKeySequence('Shift+D'), self
        )
        self.sc_clear_disappear.activated.connect(self.clearBreakpointAtExit)
        # Jump in frames
        self.sc_goto = qw.QShortcut(qg.QKeySequence('G'), self)
        self.sc_goto.activated.connect(self.gotoFrameDialog)

        self.sc_jump_bp = qw.QShortcut(qg.QKeySequence('J'), self)
        self.sc_jump_bp.activated.connect(self.jumpToBreakpoint)
        self.sc_jump_fwd = qw.QShortcut(
            qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_PageDown), self
        )
        self.sc_jump_fwd.activated.connect(self.jumpForward)
        self.sc_jump_back = qw.QShortcut(
            qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_PageUp), self
        )
        self.sc_jump_back.activated.connect(self.jumpBackward)
        self.sc_jump_nextnew = qw.QShortcut(qg.QKeySequence('N'), self)
        self.sc_jump_nextnew.activated.connect(self.jumpNextNew)
        self.sc_jump_prevnew = qw.QShortcut(qg.QKeySequence('P'), self)
        self.sc_jump_prevnew.activated.connect(self.jumpPrevNew)
        self.sc_jump_next_jump = qw.QShortcut(
            qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_N), self
        )
        self.sc_jump_next_jump.activated.connect(self.jumpNextJump)
        self.sc_jump_prev_jump = qw.QShortcut(
            qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_P), self
        )
        self.sc_jump_prev_jump.activated.connect(self.jumpPrevJump)

        self.sc_jump_nextchange = qw.QShortcut(qg.QKeySequence('C'), self)
        self.sc_jump_nextchange.activated.connect(self.jumpNextChange)
        self.sc_jump_prevchange = qw.QShortcut(
            qg.QKeySequence('Shift+C'), self
        )
        self.sc_jump_prevchange.activated.connect(self.jumpPrevChange)

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
        self.sc_hist.activated.connect(self.toggleShowTrackHistory)
        self.sc_keepsel = qw.QShortcut(qg.QKeySequence('S'), self)
        self.sc_keepsel.activated.connect(self.keepSelectionAction.toggle)
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
            qg.QKeySequence(qc.Qt.SHIFT + qc.Qt.Key_Delete), self
        )
        self.sc_remove.activated.connect(self.deleteSelectedCur)

        self.sc_remove_cur_2 = qw.QShortcut(qg.QKeySequence('Shift+X'), self)
        self.sc_remove_cur_2.activated.connect(self.deleteSelectedCur)

        self.sc_remove_range = qw.QShortcut(qg.QKeySequence('Alt+X'), self)
        self.sc_remove_range.activated.connect(self.deleteSelectedRange)
        self.sc_remove_range_2 = qw.QShortcut(
            qg.QKeySequence(qc.Qt.ALT + qc.Qt.Key_Delete), self
        )
        self.sc_remove_range_2.activated.connect(self.deleteSelectedRange)

        self.sc_rename = qw.QShortcut(qg.QKeySequence('R'), self)
        self.sc_rename.activated.connect(self.renameTrack)
        self.sc_rename_cur = qw.QShortcut(qg.QKeySequence('Shift+R'), self)
        self.sc_rename_cur.activated.connect(self.renameTrackCur)
        self.sc_speedup = qw.QShortcut(
            qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_Up), self
        )
        self.sc_speedup.activated.connect(self.speedUp)
        self.sc_slowdown = qw.QShortcut(
            qg.QKeySequence(qc.Qt.CTRL + qc.Qt.Key_Down), self
        )
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

    def deleteSelected(self, cur=False, range=False) -> None:
        widget = qw.QApplication.focusWidget()
        if isinstance(widget, TrackList):
            items = widget.selectedItems()
        else:
            return
        if len(items) == 0:
            return
        selected = [int(item.text()) for item in items]
        self.rightView.scene().setSelected(selected)
        self.rightView.scene().removeSelected()
        endFrame = -1
        if range:
            endFrame, ok = qw.QInputDialog.getInt(
                self,
                'Delete in frame range',
                'Delete till frame',
                self.cur['pos'],
                self.cur['pos'],
                2**31 - 1,
            )
            if not ok:
                return
        if cur:
            endFrame = self.cur['pos']
        for sel in selected:
            self.trackReader.deleteTrack(self.cur['pos'], sel, endFrame)
            if sel not in self.cur['tracks']:
                continue
            self.cur['tracks'].pop(sel)
            right_items = self.right_list.findItems(
                items[0].text(), qc.Qt.MatchExactly
            )
            for item in right_items:
                self.right_list.takeItem(self.right_list.row(item))
        self.sigDataFile.emit(self.track_filename, True)

    @qc.pyqtSlot()
    def deleteSelectedFut(self) -> None:
        self.deleteSelected(cur=False)

    @qc.pyqtSlot()
    def deleteSelectedCur(self) -> None:
        self.deleteSelected(cur=True)

    @qc.pyqtSlot()
    def deleteSelectedRange(self) -> None:
        self.deleteSelected(cur=False, range=True)

    @qc.pyqtSlot()
    def renameTrack(self):
        target = self.right_list.selectedItems()
        if len(target) == 0:
            return
        tid = int(target[0].text())
        val, ok = qw.QInputDialog.getInt(
            self, 'Rename track', 'New track id:', value=tid
        )
        if ok:
            print(f'Renaming track {tid} to {val}')
            self.mapTracks(val, tid, -1, False)

    @qc.pyqtSlot()
    def renameTrackCur(self):
        target = self.right_list.selectedItems()
        if len(target) == 0:
            return
        tid = int(target[0].text())
        val, ok = qw.QInputDialog.getInt(
            self, 'Rename track', 'New track id', value=tid
        )
        if ok:
            print(f'Renaming track {tid} to {val}')
            self.mapTracks(val, tid, self.cur['pos'], False)

    @qc.pyqtSlot()
    def swapTracks(self):
        source = self.all_list.selectedItems()
        target = self.right_list.selectedItems()
        if len(source) == 0 or len(target) == 0:
            return
        self.mapTracks(int(source[0].text()), int(target[0].text()), -1, True)

    @qc.pyqtSlot()
    def replaceTrack(self):
        source = self.all_list.selectedItems()
        target = self.right_list.selectedItems()
        if len(source) == 0 or len(target) == 0:
            return
        self.mapTracks(int(source[0].text()), int(target[0].text()), -1, False)

    @qc.pyqtSlot()
    def swapTracksCur(self):
        source = self.all_list.selectedItems()
        target = self.right_list.selectedItems()
        if len(source) == 0 or len(target) == 0:
            return
        self.mapTracks(
            int(source[0].text()), int(target[0].text()), self.cur['pos'], True
        )

    @qc.pyqtSlot()
    def swapTracksRange(self):
        source = self.all_list.selectedItems()
        target = self.right_list.selectedItems()
        if len(source) == 0 or len(target) == 0:
            return
        endFrame, accept = qw.QInputDialog.getInt(
            self,
            'Frame range',
            'Apply till frame',
            self.cur['pos'],
            self.cur['pos'],
            2**31 - 1,
        )
        if not accept:
            return
        self.mapTracks(
            int(source[0].text()), int(target[0].text()), endFrame, True
        )

    @qc.pyqtSlot()
    def replaceTrackCur(self):
        source = self.all_list.selectedItems()
        target = self.right_list.selectedItems()
        if len(source) == 0 or len(target) == 0:
            return
        self.mapTracks(
            int(source[0].text()), int(target[0].text()), self.cur['pos'], True
        )

    @qc.pyqtSlot()
    def replaceTracksRange(self):
        source = self.all_list.selectedItems()
        target = self.right_list.selectedItems()
        if len(source) == 0 or len(target) == 0:
            return
        endFrame, accept = qw.QInputDialog.getInt(
            self,
            'Frame range',
            'Apply till frame',
            self.cur['pos'],
            self.cur['pos'],
            2**31 - 1,
        )
        if not accept:
            return
        self.mapTracks(
            int(source[0].text()), int(target[0].text()), endFrame, False
        )

    @qc.pyqtSlot(int)
    def gotoFrame(self, frame_no):
        if frame_no < 0:
            frame_no = 0
        if self.trackReader is None or self.video_reader is None:  # or \
            #                frame_no > self.trackReader.last_frame:
            return
        if self.disableSeekAction.isChecked():
            self.sigNextFrame.emit()
        else:
            self.sigGotoFrame.emit(frame_no)

    @qc.pyqtSlot()
    def gotoFrameDialog(self):
        val, ok = qw.QInputDialog.getInt(
            self, 'Goto frame', 'Jump to frame #', value=self.cur['pos'], min=0
        )
        if ok and val >= 0:
            self.gotoFrame(val - 1)
            self.gotoFrame(val)

    @qc.pyqtSlot()
    def jumpToBreakpoint(self):
        if self.breakpoint >= 0 and not self.disableSeekAction.isChecked():
            self.gotoFrame(self.breakpoint - 1)
            self.gotoFrame(self.breakpoint)

    @qc.pyqtSlot()
    def jumpForward(self):
        target = self.cur['pos'] + self.jump_step
        if (self.cur['pos'] < self.breakpoint) and (target > self.breakpoint):
            target = self.breakpoint
        self.gotoFrame(target - 1)
        self.gotoFrame(target)

    @qc.pyqtSlot()
    def jumpBackward(self):
        target = self.cur['pos'] - self.jump_step
        if (self.cur['pos'] > self.breakpoint) and (target < self.breakpoint):
            target = self.breakpoint
        self.gotoFrame(target - 1)
        self.gotoFrame(target)

    @qc.pyqtSlot()
    def jumpNextNew(self):
        if self.trackReader is None:
            return
        frame = self.trackReader.getFrameNextNew(self.cur['pos'])
        self.gotoFrame(frame - 1)
        self.gotoFrame(frame)
        if set(self.cur['tracks']) == set(self.prev['tracks']):
            self.sigDiffMessage.emit(
                f'{self.prev["pos"]} - {self.cur["pos"]}'
                ': tracks differ but filtered out'
            )

    @qc.pyqtSlot()
    def jumpPrevNew(self):
        if self.trackReader is None:
            return
        frame = self.trackReader.getFramePrevNew(self.cur['pos'])
        self.gotoFrame(frame - 1)
        self.gotoFrame(frame)
        if set(self.cur['tracks']) == set(self.prev['tracks']):
            self.sigDiffMessage.emit(
                f'{self.prev["pos"]} - {self.cur["pos"]}'
                ': tracks differ but filtered out'
            )

    @qc.pyqtSlot()
    def jumpNextJump(self):
        if self.trackReader is None:
            return
        jumpThreshold = settings.value('review/jumpthreshold', 50.0, type=float)        
        frame, tracks = self.trackReader.getFrameNextJump(self.cur['pos'], jumpThreshold)
        self.gotoFrame(frame - 1)
        self.gotoFrame(frame)
        self.sigDiffMessage.emit(
            f'{self.prev["pos"]} - {self.cur["pos"]} Jump in: {tracks}'
            )

    @qc.pyqtSlot()
    def jumpPrevJump(self):
        if self.trackReader is None:
            return
        jumpThreshold = settings.value('review/jumpthreshold', 50.0, type=float)        
        frame, tracks = self.trackReader.getFramePrevJump(self.cur['pos'], jumpThreshold)
        self.gotoFrame(frame - 1)
        self.gotoFrame(frame)
        self.sigDiffMessage.emit(
            f'{self.prev["pos"]} - {self.cur["pos"]} Jump in tracks: {tracks}'
            )

    @qc.pyqtSlot()
    def jumpNextChange(self):
        if self.trackReader is None:
            return
        change_frames = [
            change.frame
            for change in self.trackReader.changeList
            if change.frame not in self.trackReader.undone_changes
        ]
        pos = np.searchsorted(change_frames, self.cur['pos'], side='right')
        if 0 <= pos < len(change_frames):
            # this ensures we have two successive frames on left and right pane
            self.gotoFrame(change_frames[pos] - 1)
            self.gotoFrame(change_frames[pos])

    @qc.pyqtSlot()
    def jumpPrevChange(self):
        if self.trackReader is None:
            return
        change_frames = [
            change.frame
            for change in self.trackReader.changeList
            if change.frame not in self.trackReader.undone_changes
        ]
        pos = np.searchsorted(change_frames, self.cur['pos'], side='left') - 1
        if pos < 0:
            return
        # this ensures we have two successive frames on left and right pane
        self.gotoFrame(change_frames[pos] - 1)
        self.gotoFrame(change_frames[pos])

    @qc.pyqtSlot()
    def gotoEditedPos(self):
        frame_no = int(self.pos_spin.text())
        # this ensures we have two successive frames on left and right pane
        self.gotoFrame(frame_no - 1)
        self.gotoFrame(frame_no)

    @qc.pyqtSlot()
    def nextFrame(self):
        self.gotoFrame(self.cur['pos'] + 1)
        if self.play_button.isChecked():
            self.timer.start(int(self.frame_interval / self.speed))

    @qc.pyqtSlot()
    def prevFrame(self):
        # Since going backwards changes the left frame, and keeps cur to one
        # frame ahead, we move two frames back from cur pos
        pos = self.cur['pos']
        self.gotoFrame(pos - 2)
        self.gotoFrame(pos - 1)

    def _flag_tracks(self, all_tracks, cur_tracks):
        """Change the track info to include its age, if age is more than
        `history_length` then remove this track from all tracks -
        this avoids cluttering the view with very old tracks that have not been
        seen in a long time."""
        ret = {}
        for tid, rect in all_tracks.items():
            if self.cur['pos'] - rect[4] <= self.history_length:
                ret[tid] = rect.copy()
        for tid, rect in cur_tracks.items():
            ret[tid] = np.array(rect)
        return ret

    @qc.pyqtSlot()
    @qc.pyqtSlot(str, bool)
    def updateChangeList(self, filepath='', checked=True):
        """Catches `self.sigDataFile` to update the changelist"""
        if self.trackReader is None:
            return
        change_list = [
            change
            for change in self.trackReader.changeList
            if change.frame not in self.trackReader.undone_changes
        ]
        self.changelist_widget.setChangeList(change_list)

    @qc.pyqtSlot()
    def saveChangeList(self):
        if self.trackReader is None:
            return
        fname, _ = qw.QFileDialog.getSaveFileName(
            self, 'Save list of changes', filter='Text (*.csv)'
        )
        if len(fname) > 0:
            self.trackReader.saveChangeList(fname)

    @qc.pyqtSlot()
    def loadChangeList(self):
        if self.trackReader is None:
            return
        fname, _ = qw.QFileDialog.getOpenFileName(
            self, 'Load list of changes', filter='Text (*.csv *.txt)'
        )
        if len(fname) > 0:
            self.trackReader.loadChangeList(fname)

    @qc.pyqtSlot(bool)
    def toggleSideBySideView(self, checked: bool) -> None:
        self.leftView.setVisible(checked)
        settings.setValue('review/sidebyside', checked)
        if checked:
            try:
                self.sigProjectTrackHistLeft.disconnect(
                    self.rightView.frameScene.showTrackHist
                )
            except Exception:
                pass
            try:
                self.left_list.sigSelected.disconnect(
                    self.rightView.sigSelected
                )
            except Exception:
                pass
            try:
                self.all_list.sigSelected.disconnect(
                    self.rightView.sigSelected
                )
            except Exception:
                pass

        else:
            self.sigProjectTrackHistLeft.connect(
                self.rightView.frameScene.showTrackHist
            )
            self.left_list.sigSelected.connect(self.rightView.sigSelected)
            self.all_list.sigSelected.connect(self.rightView.sigSelected)

    # @qc.pyqtSlot(qg.QPolygonF)
    # def setRoi(self, roi: qg.QPolygonF) -> None:
    #     self.roi = roi
    #
    # @qc.pyqtSlot()
    # def resetRoi(self) -> None:
    #     self.roi = None

    @qc.pyqtSlot()
    def undoCurrentChanges(self):
        self.sigUndoCurrentChanges.emit(self.cur['pos'])

    @qc.pyqtSlot(np.ndarray, int)
    def setFrame(self, frame: np.ndarray, pos: int) -> None:
        """Set the current frame, called by signal from video reader.
        This assumes we are moving forward sequentially.
        So call twice when jumping around."""
        logging.debug(f'Received frame: {pos}')
        if pos == self.cur['pos']:
            return
        tracks = self.trackReader.getTracks(pos)
        flagged_tracks = self._flag_tracks({}, tracks)
        # ts = time.perf_counter_ns()
        # if self.roi is not None:
        #     # flag tracks outside ROI
        #     keys = list(tracks.keys())
        #     for tid in keys:
        #         bbox = qg.QPolygonF(qc.QRectF(*tracks[tid][:4]))
        #         if not self.roi.intersects(bbox):
        #             self.trackReader.deleteTrack(pos, tid)
        #             tracks.pop(tid)

        if pos > 0:
            self.prev = self.cur
        else:
            self.prev = {
                'frame': np.zeros(frame.shape, dtype=frame.dtype),
                'tracks': {},
                'pos': -1,
            }
        self.cur = {'frame': frame, 'tracks': flagged_tracks, 'pos': pos}
        # When overlay is checked, show the diff on the right
        if (
            self.overlayAction.isChecked()
            and (self.prev['frame'] is not None)
            and len(frame.shape) == 3
        ):
            diff = np.zeros(
                (frame.shape[0], frame.shape[1], 3), dtype=np.uint8
            )
            diff[:, :, 1] = 0
            diff[:, :, 0] = cv2.cvtColor(self.cur['frame'], cv2.COLOR_BGR2GRAY)
            diff[:, :, 2] = cv2.cvtColor(
                self.prev['frame'], cv2.COLOR_BGR2GRAY
            )
            if self.invertOverlayColorAction.isChecked():
                diff = 255 - diff
            self.sigRightFrame.emit(diff, self.cur['pos'])
        else:
            self.sigRightFrame.emit(self.cur['frame'], self.cur['pos'])

        self.sigLeftFrame.emit(self.prev['frame'], self.prev['pos'])

        self.slider.blockSignals(True)
        self.slider.setValue(self.cur['pos'])
        self.slider.blockSignals(False)
        self.pos_spin.blockSignals(True)
        self.pos_spin.setValue(self.cur['pos'])
        self.pos_spin.blockSignals(False)

        # te = time.perf_counter_ns()
        # print(f'Time with roi "{self.roi}": {(te - ts) * 1e-6} ms')
        self.old_all_tracks = self.all_tracks.copy()
        self.all_tracks = self._flag_tracks(self.all_tracks, tracks)
        self.sigAllTracksList.emit(list(self.all_tracks.keys()))
        # If showing old tracks, display all trak ids
        if self.showOldTracksAction.isChecked():
            self.sigLeftTracks.emit(self.old_all_tracks)
            self.sigRightTracks.emit(self.all_tracks)
        else:
            self.sigLeftTracks.emit(self.prev['tracks'])
            self.sigRightTracks.emit(self.cur['tracks'])
        self.sigLeftTrackList.emit(list(self.prev['tracks'].keys()))
        self.sigRightTrackList.emit(list(self.cur['tracks'].keys()))
        self.breakpointMessage(pos)
        self.entryExitMessage(self.prev['tracks'], self.cur['tracks'])
        message = self._get_diff(self.showNewAction.isChecked())
        if len(message) > 0:
            if (
                self.showDifferenceAction.isChecked()
                or self.showNewAction.isChecked()
            ):
                self.play_button.setChecked(False)
                self.playVideo(False)
                qw.QMessageBox.information(self, 'Track mismatch', message)
            self.sigDiffMessage.emit(message)

    def _get_diff(self, show_new):
        right_keys = set(self.cur['tracks'].keys())
        all_keys = set(self.old_all_tracks.keys())
        new = right_keys - all_keys
        if show_new:
            if len(new) > 0:
                return f'Frame {self.cur["pos"] - 1}-{self.cur["pos"]}: New track on right: {new}.'
            return ''
        left_keys = set(self.prev['tracks'].keys())
        if left_keys != right_keys:
            # logging.info(f'Tracks don\'t match between frames {self.frame_no - 1} '
            #              f'and {self.frame_no}: '
            #              f'{left_keys.symmetric_difference(right_keys)}')
            left_only = left_keys - right_keys
            left_message = (
                f'Tracks only on left: {left_only}.'
                if len(left_only) > 0
                else ''
            )
            right_only = right_keys - left_keys
            right_message = (
                f'Tracks only on right: {right_only}.'
                if len(right_only) > 0
                else ''
            )
            if len(new) > 0:
                right_message += f'New tracks: <b>{new}</b>'
            return (
                f'Frame {self.cur["pos"] - 1}-{self.cur["pos"]}:'
                f'{left_message} {right_message}'
            )
        else:
            return ''

    def _setJumpThreshold(self):
        """Set the threshold distance in positions between successive frames to flag it as a jump"""
        jumpThreshold = settings.value('review/jumpthreshold', 50.0, type=float)
        jumpThreshold, accept = qw.QInputDialog.getDouble(
            self,
            'Threshold displacement for jumps',
            'Displacement', jumpThreshold, min=0.1
        )
        if not accept:
            return
        settings.setValue('review/jumpthreshold', jumpThreshold)
        
    def _setPathCmap(self, side):
        cmap, accept = qw.QInputDialog.getItem(
            self,
            'Select colormap',
            'Colormap',
            [
                'jet',
                'viridis',
                'rainbow',
                'autumn',
                'summer',
                'winter',
                'spring',
                'cool',
                'hot',
            ],
        )
        logging.debug(f'Setting colormap to {cmap}')
        if not accept:
            return
        self.pathCmap[side] = cmap
        settings.setValue(f'review/path_cmap_{side}', cmap)

    @qc.pyqtSlot()
    def setPathCmapLeft(self):
        self._setPathCmap('left')

    @qc.pyqtSlot()
    def setPathCmapRight(self):
        self._setPathCmap('right')

    @qc.pyqtSlot()
    def openTrackedData(self):
        datadir = settings.value('data/directory', '.')
        track_filename, filter = qw.QFileDialog.getOpenFileName(
            self,
            'Open tracked data',
            datadir,
            filter='HDF5 (*.h5 *.hdf);; Text (*.csv *.txt);; All files (*)',
        )
        logging.debug(f'filename:{track_filename}\nselected filter:{filter}')
        if len(track_filename) == 0:
            return
        viddir = os.path.dirname(track_filename)
        vid_filename, vfilter = qw.QFileDialog.getOpenFileName(
            self,
            f'Open video for {os.path.basename(track_filename)}',
            viddir,
            filter='Video (*.avi *.mp4 *.mpg *.mpeg *.ogg *.webm *.wmv *.mov);;'
            ' All files (*)',
        )
        logging.debug(f'filename:{vid_filename}\nselected filter:{vfilter}')
        if len(vid_filename) == 0:
            return
        # fmt = 'csv' if filter.startswith('Text') else 'hdf'
        self.setupReading(vid_filename, track_filename)

    def setupReading(self, video_path, data_path):
        try:
            self.video_reader = VideoReader(video_path, self._wait_cond)
        except IOError as e:
            qw.QMessageBox.critical(
                self,
                'Error opening video',
                f'Could not open video: {video_path}\n' f'{e}',
            )
            return False
        self.trackReader = TrackReader(data_path)
        self.video_filename = video_path
        self.track_filename = data_path
        settings.setValue(
            'data/directory', os.path.dirname(self.track_filename)
        )
        settings.setValue(
            'video/directory', os.path.dirname(self.video_filename)
        )
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
        self.prev = {'frame': None, 'tracks': None, 'pos': -1}
        self.cur = {'frame': None, 'tracks': None, 'pos': -1}
        self.history_length = self.trackReader.last_frame
        self.leftView.frameScene.setHistGradient(self.history_length)
        self.rightView.frameScene.setHistGradient(self.history_length)
        self.rightView.resetArenaAction.trigger()
        self.lim_win.connectLimitSignals(
            wminSlot=self.trackReader.setWmin,
            wmaxSlot=self.trackReader.setWmax,
            hminSlot=self.trackReader.setHmin,
            hmaxSlot=self.trackReader.setHmax,
        )
        self.rightView.sigArena.connect(self.trackReader.setRoi)
        self.rightView.resetArenaAction.triggered.connect(
            self.trackReader.resetRoi
        )
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
        self.pos_spin.setRange(0, self.trackReader.last_frame)
        self.pos_spin.blockSignals(False)
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.trackReader.last_frame)
        self.slider.blockSignals(False)
        self.sigChangeTrack.connect(self.trackReader.changeTrack)
        self.trackReader.sigChangeList.connect(
            self.changelist_widget.setChangeList
        )
        self.trackReader.sigEnd.connect(self.trackEnd)
        self.sigUndoCurrentChanges.connect(self.trackReader.undoChangeTrack)
        self.sigDataFile.emit(self.track_filename, False)
        self.gotoFrame(0)
        self.updateGeometry()
        self.tieViews(self.tieViewsAction.isChecked())
        return True

    @qc.pyqtSlot()
    def saveReviewedTracks(self):
        self.playVideo(False)
        datadir = settings.value('data/directory', '.')
        default_file = (
            datadir if self.track_filename is None else self.track_filename
        )
        track_filename, filter = qw.QFileDialog.getSaveFileName(
            self,
            'Save reviewed data',
            default_file,
            filter='HDF5 (*.h5 *.hdf);; Text (*.csv)',
        )
        logging.debug(f'filename:{track_filename}\nselected filter:{filter}')
        if len(track_filename) > 0:
            if self.save_indicator is None:
                self.save_indicator = qw.QProgressDialog(
                    'Saving track data',
                    None,
                    0,
                    self.trackReader.last_frame + 1,
                    self,
                )

                self.save_indicator.setWindowModality(qc.Qt.WindowModal)
                self.save_indicator.resize(400, 200)
                # self.trackReader.sigSavedFrames.connect(self.save_indicator.setValue)
            else:
                self.save_indicator.setRange(
                    0, self.trackReader.last_frame + 1
                )
                self.save_indicator.setValue(0)
            try:  # make sure same track reader is not connected multiple times
                self.trackReader.sigSavedFrames.disconnect()
            except TypeError:
                pass
            self.trackReader.sigSavedFrames.connect(
                self.save_indicator.setValue
            )
            self.save_indicator.show()
            try:
                self.trackReader.saveChanges(track_filename)
                self.trackReader.data_path = track_filename
                self.track_filename = track_filename
                self.sigDataFile.emit(track_filename, False)
                self.save_indicator.setValue(self.trackReader.last_frame + 1)
            except OSError as err:
                qw.QMessageBox.critical(
                    self,
                    'Error opening file for writing',
                    f'File {track_filename} could not be opened.\n{err}',
                )

    @qc.pyqtSlot()
    def doQuit(self):
        # self._wait_cond.set()
        self.vid_info.close()
        self.changelist_widget.close()
        if (
            self.trackReader is not None
            and len(self.trackReader.changeList) > 0
        ):
            self.saveReviewedTracks()
        diff = 0
        if self.showNewAction.isChecked():
            diff = 1
        elif self.showDifferenceAction.isChecked():
            diff = 2
        settings.setValue('review/showdiff', diff)
        settings.setValue(
            'review/disable_seek', self.disableSeekAction.isChecked()
        )
        self.sigQuit.emit()

    @qc.pyqtSlot(bool)
    def playVideo(self, play: bool):
        if self.video_reader is None:
            return
        if play:
            self.play_button.setText('Pause (Space)')
            self.play_button.setIcon(
                self.style().standardIcon(qw.QStyle.SP_MediaPause)
            )
            self.playAction.setText('Pause (Space)')
            self.playAction.setIcon(
                self.style().standardIcon(qw.QStyle.SP_MediaPause)
            )
            self.timer.start(int(self.frame_interval / self.speed))
        else:
            self.play_button.setText('Play (Space)')
            self.play_button.setIcon(
                self.style().standardIcon(qw.QStyle.SP_MediaPlay)
            )
            self.playAction.setText('Play (Space)')
            self.playAction.setIcon(
                self.style().standardIcon(qw.QStyle.SP_MediaPlay)
            )
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
        print('Reset video ...')
        if self.video_reader is None:
            # Not initialized - do nothing
            return
        if len(self.trackReader.changeList) > 0:
            ret = qw.QMessageBox.question(
                self,
                'Confirm reset',
                'This will reset all changes!'
                '\nTo save your work, press No and save data first',
                qw.QMessageBox.Yes | qw.QMessageBox.No,
            )
            if ret == qw.QMessageBox.No:
                return
        self._wait_cond.set()
        self.playVideo(False)
        self.play_button.setChecked(False)
        self.leftView.clearAll()
        self.rightView.clearAll()
        self.setupReading(self.video_filename, self.track_filename)

    @qc.pyqtSlot(int, int, int, bool)
    def mapTracks(
        self, newId: int, origId: int, endFrame: int, swap: bool
    ) -> None:
        """Map newId to origId, up to and including endFrame.
        If swap is True, do a swap, otherwise assign.
        """
        if newId == origId:
            return
        if swap:
            self.trackReader.swapTrack(
                self.cur['pos'], origId, newId, endFrame
            )
        else:
            self.trackReader.changeTrack(
                self.cur['pos'], origId, newId, endFrame
            )
        tracks = self.trackReader.getTracks(self.cur['pos'])
        self.sigRightTrackList.emit(list(tracks.keys()))
        self.cur['tracks'] = self._flag_tracks({}, tracks)
        self.sigRightTracks.emit(self.cur['tracks'])
        self.sigDataFile.emit(self.track_filename, True)

    @qc.pyqtSlot()
    def videoEnd(self):
        self.playVideo(False)
        self.play_button.setChecked(False)
        qw.QMessageBox.information(
            self, 'Finished processing', 'End of video reached.'
        )

    @qc.pyqtSlot()
    def trackEnd(self):
        self.playVideo(False)
        self.play_button.setChecked(False)
        qw.QMessageBox.information(
            self, 'Finished processing', 'End of tracks reached reached.'
        )


class ReviewerMain(qw.QMainWindow):
    sigQuit = qc.pyqtSignal()

    def __init__(self):
        super(ReviewerMain, self).__init__()
        self.reviewWidget = ReviewWidget()
        self.rightDock = qw.QDockWidget('Changes')
        self.rightDock.setFeatures(
            qw.QDockWidget.DockWidgetMovable
            | qw.QDockWidget.DockWidgetFloatable
        )
        self.rightDock.setWidget(self.reviewWidget.changelist_widget)
        self.rightDock.setVisible(False)
        self.reviewWidget.showChangeListAction.triggered.connect(
            self.rightDock.setVisible
        )
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self.rightDock)

        fileMenu = self.menuBar().addMenu('&File')
        fileMenu.addAction(self.reviewWidget.openAction)
        fileMenu.addAction(self.reviewWidget.saveAction)
        fileMenu.addAction(self.reviewWidget.loadChangeListAction)
        fileMenu.addAction(self.reviewWidget.saveChangeListAction)

        self.sc_quit = qw.QShortcut(qg.QKeySequence('Ctrl+Q'), self)
        self.sc_quit.activated.connect(self.close)
        self.quitAction = qw.QAction('Quit (Ctrl+Q)')
        self.quitAction.triggered.connect(self.close)
        fileMenu.addAction(self.quitAction)

        diffMenu = self.menuBar().addMenu('&Diff settings')
        diffMenu.addAction(self.reviewWidget.overlayAction)
        diffMenu.addAction(self.reviewWidget.invertOverlayColorAction)
        diffMenu.addAction(self.reviewWidget.toggleSideBySideAction)
        diffgrp = qw.QActionGroup(self)
        diffgrp.addAction(self.reviewWidget.showDifferenceAction)
        diffgrp.addAction(self.reviewWidget.showNewAction)
        diffgrp.addAction(self.reviewWidget.showNoneAction)
        diffgrp.addAction(self.reviewWidget.setJumpThresholdAction)
        diffgrp.setExclusive(True)
        diffMenu.addActions(diffgrp.actions())

        viewMenu = self.menuBar().addMenu('&View')
        viewMenu.addAction(self.reviewWidget.tieViewsAction)
        viewMenu.addAction(self.reviewWidget.showGrayscaleAction)
        viewMenu.addAction(self.reviewWidget.setColorAction)
        viewMenu.addAction(self.reviewWidget.setSelectedColorAction)
        viewMenu.addAction(self.reviewWidget.setAlphaUnselectedAction)
        viewMenu.addAction(self.reviewWidget.autoColorAction)
        viewMenu.addAction(self.reviewWidget.colormapAction)
        viewMenu.addAction(self.reviewWidget.pathCmapLeftAction)
        viewMenu.addAction(self.reviewWidget.pathCmapRightAction)
        viewMenu.addAction(self.reviewWidget.keepSelectionAction)
        viewMenu.addSeparator()
        viewMenu.addAction(self.reviewWidget.fontSizeAction)
        viewMenu.addAction(self.reviewWidget.relativeFontSizeAction)
        viewMenu.addAction(self.reviewWidget.lineWidthAction)
        viewMenu.addAction(self.reviewWidget.setPathDiaAction)
        viewMenu.addAction(self.reviewWidget.setMarkerThicknessAction)
        viewMenu.addSeparator()
        viewMenu.addAction(self.reviewWidget.showIdAction)
        viewMenu.addAction(self.reviewWidget.showBboxAction)
        viewMenu.addAction(self.reviewWidget.labelInsideAction)
        viewMenu.addAction(self.reviewWidget.showOldTracksAction)
        viewMenu.addAction(self.reviewWidget.showHistoryAction)
        viewMenu.addSeparator()
        viewMenu.addAction(self.reviewWidget.showLimitsAction)
        viewMenu.addAction(self.reviewWidget.histlenAction)
        viewMenu.addAction(self.reviewWidget.histGradientAction)

        viewMenu.addAction(self.reviewWidget.showChangeListAction)
        themeGroup = qw.QActionGroup(self.reviewWidget)
        themeGroup.setExclusive(True)
        selected = settings.value('theme', 'dark')
        for theme in STYLE_SHEETS:
            action = themeGroup.addAction(theme.capitalize())
            action.setCheckable(True)
            action.triggered.connect(
                lambda chk, name=theme: ut.setStyleSheet(name)
            )
            if theme == selected:
                action.setChecked(True)
        ut.setStyleSheet(selected)
        themeMenu = viewMenu.addMenu('&Theme')
        themeMenu.addActions(themeGroup.actions())

        zoomMenu = self.menuBar().addMenu('&Zoom')
        zoomMenu.addAction(self.reviewWidget.zoomInLeftAction)
        zoomMenu.addAction(self.reviewWidget.zoomInRightAction)
        zoomMenu.addAction(self.reviewWidget.zoomOutLeftAction)
        zoomMenu.addAction(self.reviewWidget.zoomOutRightAction)

        playMenu = self.menuBar().addMenu('&Play')
        playMenu.addAction(self.reviewWidget.disableSeekAction)
        playMenu.addAction(self.reviewWidget.playAction)
        playMenu.addAction(self.reviewWidget.speedUpAction)
        playMenu.addAction(self.reviewWidget.slowDownAction)
        playMenu.addAction(self.reviewWidget.resetAction)

        playMenu.addSeparator()
        playMenu.addAction(self.reviewWidget.nextFrameAction)
        playMenu.addAction(self.reviewWidget.prevFrameAction)
        playMenu.addAction(self.reviewWidget.gotoFrameAction)
        playMenu.addAction(self.reviewWidget.jumpForwardAction)
        playMenu.addAction(self.reviewWidget.jumpBackwardAction)
        playMenu.addSeparator()
        playMenu.addAction(self.reviewWidget.frameBreakpointAction)
        playMenu.addAction(self.reviewWidget.curBreakpointAction)
        playMenu.addAction(self.reviewWidget.entryBreakpointAction)
        playMenu.addAction(self.reviewWidget.exitBreakpointAction)

        playMenu.addAction(self.reviewWidget.clearBreakpointAction)
        playMenu.addAction(self.reviewWidget.clearEntryBreakpointAction)
        playMenu.addAction(self.reviewWidget.clearExitBreakpointAction)
        playMenu.addSeparator()
        playMenu.addAction(self.reviewWidget.jumpToBreakpointAction)
        playMenu.addAction(self.reviewWidget.jumpNextNewAction)
        playMenu.addAction(self.reviewWidget.jumpPrevNewAction)
        playMenu.addAction(self.reviewWidget.jumpNextChangeAction)
        playMenu.addAction(self.reviewWidget.jumpPrevChangeAction)
        playMenu.addAction(self.reviewWidget.jumpNextJumpAction)
        playMenu.addAction(self.reviewWidget.jumpPrevJumpAction)

        actionMenu = self.menuBar().addMenu('Action')
        actionMenu.addActions(
            [
                self.reviewWidget.swapTracksAction,
                self.reviewWidget.swapTracksCurAction,
                self.reviewWidget.swapTracksRangeAction,
                self.reviewWidget.replaceTrackAction,
                self.reviewWidget.replaceTrackCurAction,
                self.reviewWidget.replaceTracksRangeAction,
                self.reviewWidget.renameTrackAction,
                self.reviewWidget.renameTrackCurAction,
                self.reviewWidget.deleteTrackAction,
                self.reviewWidget.deleteTrackCurAction,
            ]
        )
        actionMenu.addSeparator()
        actionMenu.addActions(
            [
                self.reviewWidget.undoCurrentChangesAction,
                self.reviewWidget.enableDrawArenaAction,
                self.reviewWidget.rightView.resetArenaAction,
            ]
        )
        self.debugAction = qw.QAction('Debug')
        self.debugAction.setToolTip('Enable debug print to console')
        self.debugAction.setCheckable(True)
        v = settings.value('review/debug', logging.INFO)
        self.setDebug(v == logging.DEBUG)
        self.debugAction.setChecked(v == logging.DEBUG)
        self.debugAction.triggered.connect(self.setDebug)
        actionMenu.addAction(self.debugAction)

        helpMenu = self.menuBar().addMenu('Help')
        aboutAction = helpMenu.addAction('&About')
        aboutAction.setIcon(
            self.style().standardIcon(
                qw.QStyle.StandardPixmap.SP_MessageBoxInformation
            )
        )
        aboutAction.triggered.connect(displayAbout)
        docAction = helpMenu.addAction('Documentation')
        docAction.setIcon(
            self.style().standardIcon(
                qw.QStyle.StandardPixmap.SP_MessageBoxQuestion
            )
        )
        docAction.triggered.connect(displayDoc)
        helpMenu.addAction(self.reviewWidget.vidinfoAction)

        toolbar = self.addToolBar('View')
        toolbar.addActions(
            [
                self.reviewWidget.zoomInLeftAction,
                self.reviewWidget.zoomInRightAction,
                self.reviewWidget.zoomOutLeftAction,
                self.reviewWidget.zoomOutRightAction,
                self.reviewWidget.tieViewsAction,
                self.reviewWidget.showOldTracksAction,
                self.reviewWidget.showHistoryAction,
            ]
        )

        toolbar.addActions(actionMenu.actions())

        toggleToolbarAction = viewMenu.addAction('Show toolbar')
        toggleToolbarAction.setCheckable(True)
        toggleToolbarAction.triggered.connect(toolbar.setVisible)
        toggleToolbarAction.setChecked(True)

        self.reviewWidget.sigDataFile.connect(self.updateTitle)
        self.sigQuit.connect(self.reviewWidget.doQuit)
        self.statusLabel = qw.QLabel()
        self.reviewWidget.sigDiffMessage.connect(self.statusLabel.setText)
        self.statusBar().addWidget(self.statusLabel)
        self.posLabel = qw.QLabel()
        self.statusBar().addPermanentWidget(self.posLabel)
        self.reviewWidget.sigMousePosMessage.connect(self.posLabel.setText)
        # make sure to show change list at first
        self.reviewWidget.showChangeListAction.trigger()
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

    @qc.pyqtSlot(str, bool)
    def updateTitle(self, filename: str, changed: bool) -> None:
        if changed:
            self.setWindowTitle(f'* {filename}: Argos:Review')
        else:
            self.setWindowTitle(f'{filename}: Argos:Review')


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
    video_path = (
        'C:/Users/raysu/Documents/src/argos_data/dump/2020_02_20_00270.avi'
    )
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
