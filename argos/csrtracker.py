# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-16 5:05 PM

"""
===================================================================
Wrapper to use CSRT algorithm from OpenCV to track multiple objects
===================================================================

"""
import logging
from typing import Tuple
import numpy as np
import cv2
from PyQt5 import (
    QtCore as qc,
    QtWidgets as qw
)

from argos.utility import match_bboxes, pairwise_distance, init
from argos.constants import OutlineStyle, DistanceMetric

settings = init()


class CSRTracker(object):
    """Wrapper around OpenCV CSRT to maintain age information and
    reinitialization against segmentation data.

    """

    def __init__(self, frame: np.ndarray, bbox: np.ndarray, tid: int):
        self.id_ = tid
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, tuple(bbox))
        self.age = 0
        self.misses = 0

    def update(self, frame: np.ndarray) -> Tuple[int, Tuple[int]]:
        self.age += 1
        ret, bbox = self.tracker.update(frame)
        if not ret:
            self.misses += 1
        return np.array(bbox)

    def reinit(self, frame: np.ndarray, bbox: np.ndarray) -> None:
        """Reinitialize tracker to specified bounding box in frame"""
        self.tracker.init(frame, tuple(bbox))
        self.age = 0
        self.misses = 0


class CSRMultiTracker(qc.QObject):

    sigTracked = qc.pyqtSignal(dict, int)

    def __init__(self, *args, **kwargs):
        super(CSRMultiTracker, self).__init__(*args, **kwargs)
        self.checkAge = settings.value('csrt/checkAge', 10, type=int)  # check against segmentation after these many frames
        self.checkSeq = settings.value('csrt/checkSeq', 1, type=int)  # keep checking for these many frames for missing objects
        self.missLimit = settings.value('csrt/missLimit', 3, type=int)  # delete tracker after these many misses
        self.maxDist = settings.value('csrt/maxDist', 0.3, type=float)
        if settings.value('csrt/distMetric', 'iou', type=str) == 'euclidean':
            self.distMetric = DistanceMetric.euclidean
        else:
            self.distMetric = DistanceMetric.iou
        # dynamic variables
        self.trackers = {}
        self._next_id = 1
        self.age = 0
        self.check_count = 0

    def _add_tracker(self, frame: np.ndarray, bbox: np.ndarray) -> None:
        tracker = CSRTracker(frame, bbox, self._next_id)
        self.trackers[self._next_id] = tracker
        logging.debug(f'=== Added tracker {tracker.id_} for bbox: {bbox}')
        self._next_id += 1
        return tracker.id_

    def find_nonoverlapping(self, bboxes):
        """Remove entries which are within `maxDist` distance from another bbox
        considering them to be the same object detected twice."""
        dist = pairwise_distance(bboxes, bboxes,
                                 OutlineStyle.bbox,
                                 DistanceMetric.iou)
        close_row, close_col = np.where(dist <= self.maxDist)
        ignore = close_col[close_col > close_row]
        if len(ignore) > 0:
            logging.debug(f'Ignore {ignore}')
        valid_idx = set(list(range(bboxes.shape[0]))) - set(ignore)
        logging.debug(f'Valid indices: {valid_idx}')
        return bboxes[list(valid_idx)].copy()

    @qc.pyqtSlot(np.ndarray, np.ndarray, int)
    def track(self, frame: np.ndarray, bboxes: np.ndarray, pos: int) -> None:
        """Track objects in frame, possibly comparing them to bounding boxes
        in ``bboxes``

        This uses a hybrid of CSRT with Hungarian algorithm.

        Parameters
        ----------
        frame: np.ndarray
            Image in which objects are to be tracked.
        bboxes: np.ndarray
            Array of bounding boxes from segmentation. In case there are no
            existing trackers (first call to this function), then initialize
            one tracker for each bounding box.
        pos: int
            Frame position in video. Not used directly, but emitted with the
            results via ``sigTracked`` so downstream methods can relate to the
            frame.
        """
        # Now do the checks for trackers that are off target

        self.age += 1
        if len(self.trackers) == 0 and bboxes.size > 0:
            valid = self.find_nonoverlapping(bboxes)
            ret = {self._add_tracker(frame, valid[ii]): valid[ii]
                   for ii in range(valid.shape[0])}
            logging.debug(f'==== Added initial trackers \n{ret}')
            self.sigTracked.emit(ret, pos)
            return

        predicted = {id_: tracker.update(frame)
                     for id_, tracker in self.trackers.items()}
        self.sigTracked.emit(predicted, pos)
        if self.age > self.checkAge:
            valid = self.find_nonoverlapping(bboxes)
            matched, new_unmatched, old_unmatched = match_bboxes(
                predicted, valid, boxtype=OutlineStyle.bbox,
                metric=self.distMetric,
                max_dist=self.maxDist)
            logging.debug(f'==== matching bboxes Frame: {pos} ====')
            logging.debug(f'Input bboxes: {valid}\n'
                          f'Matched: {matched}\n'
                          f'New unmatched: {new_unmatched}\n'
                          f'Old unmatched: {old_unmatched}')
            # Renitialize trackers that matched to closest bounding box
            for tid, idx in matched.items():
                self.trackers[tid].reinit(frame, valid[idx])
            # Increase miss count for unmatched trackers
            for tid in old_unmatched:
                self.trackers[tid].misses += 1
            # Add trackers for bboxes that did not match any tracker
            for idx in new_unmatched:
                self._add_tracker(frame, valid[idx])
            self.check_count += 1
        # Remove the trackers that missed too many times - only after we have
        # given them `checkSeq` chances for rematching
        if self.check_count >= self.checkSeq:
            self.trackers = {tid: tracker for tid, tracker in self.trackers.items()
                             if tracker.misses < self.missLimit}
            self.age = 0
            self.check_count = 0

    @qc.pyqtSlot(int)
    def setCheckAge(self, age: int) -> None:
        self.checkAge = age

    @qc.pyqtSlot(int)
    def setCheckSeq(self, val: int) -> None:
        self.checkSeq = val

    @qc.pyqtSlot(int)
    def setMissLimit(self, val: int):
        self.missLimit = val

    @qc.pyqtSlot(float)
    def setMaxDist(self, val: float) -> None:
        self.maxDist = val

    @qc.pyqtSlot(str)
    def setDistMetric(self, metric: str) -> None:
        if metric.lower() == 'iou':
            self.distMetric = DistanceMetric.iou
        elif metric.lower() == 'euclidean':
            self.distMetric = DistanceMetric.euclidean
        else:
            raise ValueError(f'Unknown distance metric {metric}')
            

    @qc.pyqtSlot()
    def reset(self):
        self.trackers = {}
        self._next_id = 1
        self.age = 0
        self.check_count = 0


class CSRTWidget(qw.QWidget):
    """Wrapper widget providing access to CSRT parameters and to run it in a
    separate thread.

    Since this tracker requires both the image and the bounding boxes of the
    segmented objects, but upstream classes are designed to send them
    separately (frame from VideoWidget and via ``sigFrame`` and bboxes from
    any of the segmentation widgets via ``sigProcessed`` signal, we need to
    synchronize them. This is done by two slots, ``setFrame`` and ``setBboxes``
    which, track the frame position for the image and that for the bboxes
    and emits the data via ``sigTrack`` when the two match. The position
    variables are reset after that.

    """
    sigTrack = qc.pyqtSignal(np.ndarray, np.ndarray, int)
    sigTracked = qc.pyqtSignal(dict, int)
    sigQuit = qc.pyqtSignal()
    sigReset = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(CSRTWidget, self).__init__(*args, **kwargs)
        self.tracker = CSRMultiTracker()
        self._frame = None
        self._framePos = -1
        self._bboxes = None
        self._bboxesPos = -1
        layout = qw.QFormLayout()
        self._checkAgeLabel = qw.QLabel('Check after every N frames')
        self._checkAgeSpin = qw.QSpinBox()
        self._checkAgeSpin.setRange(0, 100)
        value = settings.value('csrt/checkAge', self.tracker.checkAge,
                               type=int)
        self.tracker.checkAge = value
        self._checkAgeSpin.setValue(value)
        self._checkAgeSpin.setToolTip('Verify tracked bounding boxes against'
                                        ' segmented bounding boxes every this'
                                        ' many frames. Remove or reinitialize'
                                        ' tracks that are off.')
        layout.addRow(self._checkAgeLabel, self._checkAgeSpin)
        self._checkSeqLabel = qw.QLabel('# of checks')
        self._checkSeqSpin = qw.QSpinBox()
        self._checkSeqSpin.setRange(0, 100)
        value = settings.value('csrt/checkSeq', self.tracker.checkSeq,
                               type=int)
        self._checkSeqSpin.setValue(value)
        self.tracker.checkSeq = value
        self._checkSeqSpin.setToolTip('Try this many times before removing a'
                                        ' failed tracker.')
        layout.addRow(self._checkSeqLabel, self._checkSeqSpin)
        self._missLimitLabel = qw.QLabel('Miss limit')
        self._missLimitSpin = qw.QSpinBox()
        self._missLimitSpin.setToolTip('Number of misses before a tracker is'
                                         ' removed.')
        self._missLimitSpin.setRange(1, 100)
        value = settings.value('csrt/missLimit', self.tracker.missLimit,
                               type=int)
        self._missLimitSpin.setValue(value)
        self.tracker.missLimit = value
        layout.addRow(self._missLimitLabel, self._missLimitSpin)
        self._maxDistLabel = qw.QLabel('Minimum separation')
        self._maxDistSpin = qw.QDoubleSpinBox()
        self._maxDistSpin.setToolTip('Minimum separation between a tracker and'
                                       ' its closest bounding box to consider'
                                       ' them to be separate objects.')
        self._maxDistSpin.setValue(self.tracker.maxDist)
        layout.addRow(self._maxDistLabel, self._maxDistSpin)
        self._distMetricLabel = qw.QLabel('Distance metric')
        self._distMetricCombo = qw.QComboBox(self)
        self._distMetricCombo.addItems(['IoU', 'Euclidean'])
        if self.tracker.distMetric == DistanceMetric.iou:
            self._distMetricCombo.setCurrentText('IoU')
        else:
            self._distMetricCombo.setCurrentText('Euclidean')
        self._distMetricCombo.setToolTip(
            'Distance metric to use for measuring proximity')
        layout.addRow(self._distMetricLabel, self._distMetricCombo)
        self._disableCheck = qw.QCheckBox('Disable tracking')
        self._disableCheck.setToolTip('Directly show segmentation results '
                                      'without any tracking')
        layout.addWidget(self._disableCheck)
        self.setLayout(layout)
        ################
        self.thread = qc.QThread()
        self.tracker.moveToThread(self.thread)
        self.sigTrack.connect(self.tracker.track)
        self.tracker.sigTracked.connect(self.sigTracked)
        self.sigReset.connect(self.tracker.reset)
        self.sigQuit.connect(self.saveSettings)
        self.sigQuit.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self._checkAgeSpin.valueChanged.connect(self.tracker.setCheckAge)
        self._checkSeqSpin.valueChanged.connect(self.tracker.setCheckSeq)
        self._missLimitSpin.valueChanged.connect(self.tracker.setMissLimit)
        self._maxDistSpin.valueChanged.connect(self.tracker.setMaxDist)
        self._distMetricCombo.currentTextChanged.connect(
            self.tracker.setDistMetric)
        self.thread.start()

    @qc.pyqtSlot(np.ndarray, int)
    def setFrame(self, frame: np.ndarray, pos: int) -> None:
        """Slot to store video frame and frame position, and signal the tracker
        if the bboxes for the same frame are available"""
        logging.debug(f'Received frame: {pos}')
        if self._disableCheck.isChecked():
            return
        self._frame = frame
        self._framePos = pos
        if self._framePos >= 0 and self._framePos == self._bboxesPos:
            logging.debug(f'Emitting signal for frame {self._framePos}')
            self.sigTrack.emit(self._frame, self._bboxes,
                               self._framePos)
            self._framePos = -1
            self._bboxesPos = -1

    @qc.pyqtSlot(np.ndarray, int)
    def setBboxes(self, bboxes: np.ndarray, pos: int) -> None:
        """Slot to store bounding boxes and frame position, and signal the
        tracker if the image for the same frame is available"""
        logging.debug(f'Received bboxes: {pos}')
        if self._disableCheck.isChecked():
            self.sigTracked.emit({ii: bboxes[ii]
                                  for ii in range(bboxes.shape[0])},
                                 pos)
            return
        self._bboxes = bboxes
        self._bboxesPos = pos
        if self._framePos >= 0 and self._framePos == self._bboxesPos:
            logging.debug(f'Emitting signal for frame {self._framePos}')
            self.sigTrack.emit(self._frame, self._bboxes,
                               self._framePos)
            self._framePos = -1
            self._bboxesPos = -1

    @qc.pyqtSlot()
    def saveSettings(self):
        settings.setValue('csrt/checkAge', self.tracker.checkAge)
        settings.setValue('csrt/checkSeq', self.tracker.checkSeq)
        settings.setValue('csrt/missLimit', self.tracker.missLimit)
        settings.setValue('csrt/maxDist', self.tracker.maxDist)
        settings.setValue('csrt/distMetric',
                          'iou' if self.tracker.distMetric == DistanceMetric.iou
                          else 'euclidean')
