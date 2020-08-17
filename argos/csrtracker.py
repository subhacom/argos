# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-16 5:05 PM

"""CSRT algorithm built into OpenCV"""
import logging
from typing import Tuple
import numpy as np
import cv2
from PyQt5 import (
    QtCore as qc,
    QtWidgets as qw
)

from argos.utility import match_bboxes, init
from argos.constants import OutlineStyle, DistanceMetric

settings = init()


class CSRTracker(object):
    """Wrapper around OpenCV CSRT to maintain age information and
    reinitialization against segmentation data,"""

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
        self.check_age = settings.value('csrt/check_age', 10)  # check against segmentation after these many frames
        self.check_seq = settings.value('csrt/check_seq', 1)  # keep checking for these many frames for missing objects
        self.miss_limit = settings.value('csrt/miss_limit', 3)  # delete tracker after these many misses
        self.max_dist = settings.value('csrt/max_dist', 0.3)
        if settings.value('csrt/max_dist', 'iou') == 'euclidean':
            self.dist_metric = DistanceMetric.euclidean
        else:
            self.dist_metric = DistanceMetric.iou
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
        if len(self.trackers) == 0:
            ret = {self._add_tracker(frame, bboxes[ii]): bboxes[ii].copy()
                   for ii in range(bboxes.shape[0])}
            logging.debug(f'==== Added initial trackers \n{ret}')
            self.sigTracked.emit(ret, pos)
            return

        predicted = {id_: tracker.update(frame)
                     for id_, tracker in self.trackers.items()}
        self.sigTracked.emit(predicted, pos)
        logging.debug(f'############ Predicted trackers')
        logging.debug('\n'.join([str(it) for it in predicted.items()]))
        logging.debug('\n')
        logging.debug('Received bboxes:\n'
                      f'{bboxes}')
        if self.age > self.check_age:
            matched, new_unmatched, old_unmatched = match_bboxes(
                predicted, bboxes, boxtype=OutlineStyle.bbox,
                metric=self.dist_metric,
                max_dist=self.max_dist)
            logging.info(f'==== matching bboxes Frame: {pos} ====')
            logging.info(f'Input bboxes: {bboxes}\n'
                          f'Matched: {matched}\n'
                          f'New unmatched: {new_unmatched}\n'
                          f'Old unmatched: {old_unmatched}')
            # Renitialize trackers that matched to closest bounding box
            for tid, idx in matched.items():
                self.trackers[tid].reinit(frame, bboxes[idx])
            # Increase miss count for unmatched trackers
            for tid in old_unmatched:
                self.trackers[tid].misses += 1
            # Add trackers for bboxes that did not match any tracker
            for idx in new_unmatched:
                self._add_tracker(frame, bboxes[idx])
            self.check_count += 1
        # Remove the trackers that missed too many times - only after we have
        # given them `check_seq` chances for rematching
        if self.check_count >= self.check_seq:
            self.trackers = {tid: tracker for tid, tracker in self.trackers.items()
                             if tracker.misses < self.miss_limit}
            self.age = 0
            self.check_count = 0

    @qc.pyqtSlot(int)
    def setCheckAge(self, age: int) -> None:
        self.check_age = age

    @qc.pyqtSlot(int)
    def setCheckSeq(self, val: int) -> None:
        self.check_seq = val

    @qc.pyqtSlot(int)
    def setMissLimit(self, val: int):
        self.miss_limit = val

    @qc.pyqtSlot(float)
    def setMaxDist(self, val: float) -> None:
        self.max_dist = val

    @qc.pyqtSlot(str)
    def setDistMetric(self, metric: str) -> None:
        if metric.lower() == 'iou':
            self.dist_metric = DistanceMetric.iou
        elif metric.lower() == 'euclidean':
            self.dist_metric = DistanceMetric.euclidean
        else:
            raise ValueError(f'Unknown distance metric {metric}')
            

    @qc.pyqtSlot()
    def reset(self):
        self.trackers = {}
        self._next_id = 1
        self.age = 0
        self.check_count = 0

    @qc.pyqtSlot()
    def saveSettings(self):
        settings.setValue('csrt/check_age', self.check_age)
        settings.setValue('csrt/check_seq', self.check_seq)
        settings.setValue('csrt/miss_limit', self.miss_limit)
        settings.setValue('csrt/max_dist', self.max_dist)
        settings.setValue('csrt/dist_metric',
                          'iou' if self.dist_metric == DistanceMetric.iou
                          else 'euclidean')


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
        self._frame_pos = -1
        self._bboxes = None
        self._bboxes_pos = -1
        layout = qw.QFormLayout()
        self._check_age_label = qw.QLabel('Check after every N frames')
        self._check_age_spin = qw.QSpinBox()
        self._check_age_spin.setRange(0, 100)
        value = settings.value('csrt/check_age', self.tracker.check_age,
                               type=int)
        self.tracker.check_age = value
        self._check_age_spin.setValue(value)
        self._check_age_spin.setToolTip('Verify tracked bounding boxes against'
                                        ' segmented bounding boxes every this'
                                        ' many frames. Remove or reinitialize'
                                        ' tracks that are off.')
        layout.addRow(self._check_age_label, self._check_age_spin)
        self._check_seq_label = qw.QLabel('# of checks')
        self._check_seq_spin = qw.QSpinBox()
        self._check_seq_spin.setRange(0, 100)
        value = settings.value('csrt/check_seq', self.tracker.check_seq,
                               type=int)
        self._check_seq_spin.setValue(value)
        self.tracker.check_seq = value
        self._check_seq_spin.setToolTip('Try this many times before removing a'
                                        ' failed tracker.')
        layout.addRow(self._check_seq_label, self._check_seq_spin)
        self._miss_limit_label = qw.QLabel('Miss limit')
        self._miss_limit_spin = qw.QSpinBox()
        self._miss_limit_spin.setToolTip('Number of misses before a tracker is'
                                         ' removed.')
        self._miss_limit_spin.setRange(1, 100)
        value = settings.value('csrt/miss_limit', self.tracker.miss_limit,
                               type=int)
        self._miss_limit_spin.setValue(value)
        self.tracker.miss_limit = value
        layout.addRow(self._miss_limit_label, self._miss_limit_spin)
        self._max_dist_label = qw.QLabel('Minimum separation')
        self._max_dist_spin = qw.QDoubleSpinBox()
        self._max_dist_spin.setToolTip('Minimum separation between a tracker and'
                                       ' its closest bounding box to consider'
                                       ' them to be separate objects.')
        value = settings.value('csrt/max_dist', self.tracker.max_dist,
                               type=float)
        self._max_dist_spin.setValue(value)
        self.tracker.max_dist = value
        layout.addRow(self._max_dist_label, self._max_dist_spin)
        self._dist_metric_label = qw.QLabel('Distance metric')
        self._dist_metric_combo = qw.QComboBox(self)
        self._dist_metric_combo.addItems(['IoU', 'Euclidean'])
        if self.tracker.dist_metric == DistanceMetric.iou:
            self._dist_metric_combo.setCurrentText('IoU')
        else:
            self._dist_metric_combo.setCurrentText('Euclidean')
        self._dist_metric_combo.setToolTip('Distance metric to use for measuring proximity')
        layout.addRow(self._dist_metric_label, self._dist_metric_combo)
        self._disable_check = qw.QCheckBox('Disable tracking')
        self._disable_check.setToolTip('Directly show segmentation results '
                                       'without any tracking')
        layout.addWidget(self._disable_check)
        self.setLayout(layout)
        ################
        self.thread = qc.QThread()
        self.tracker.moveToThread(self.thread)
        self.sigTrack.connect(self.tracker.track)
        self.tracker.sigTracked.connect(self.sigTracked)
        self.sigReset.connect(self.tracker.reset)
        self.sigQuit.connect(self.tracker.saveSettings)
        self.sigQuit.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self._check_age_spin.valueChanged.connect(self.tracker.setCheckAge)
        self._check_seq_spin.valueChanged.connect(self.tracker.setCheckSeq)
        self._miss_limit_spin.valueChanged.connect(self.tracker.setMissLimit)
        self._max_dist_spin.valueChanged.connect(self.tracker.setMaxDist)
        self._dist_metric_combo.currentTextChanged.connect(self.tracker.setDistMetric)
        self.thread.start()

    @qc.pyqtSlot(np.ndarray, int)
    def setFrame(self, frame: np.ndarray, pos: int) -> None:
        """Slot to store video frame and frame position, and signal the tracker
        if the bboxes for the same frame are available"""
        logging.debug(f'Received frame: {pos}')
        if self._disable_check.isChecked():
            return
        self._frame = frame
        self._frame_pos = pos
        if self._frame_pos >= 0 and self._frame_pos == self._bboxes_pos:
            logging.debug(f'Emitting signal for frame {self._frame_pos}')
            self.sigTrack.emit(self._frame, self._bboxes,
                               self._frame_pos)
            self._frame_pos = -1
            self._bboxes_pos = -1

    @qc.pyqtSlot(np.ndarray, int)
    def setBboxes(self, bboxes: np.ndarray, pos: int) -> None:
        """Slot to store bounding boxes and frame position, and signal the
        tracker if the image for the same frame is available"""
        logging.debug(f'Received bboxes: {pos}')
        if self._disable_check.isChecked():
            self.sigTracked.emit({ii: bboxes[ii]
                                  for ii in range(bboxes.shape[0])},
                                 pos)
            return
        self._bboxes = bboxes
        self._bboxes_pos = pos
        if self._frame_pos >= 0 and self._frame_pos == self._bboxes_pos:
            logging.debug(f'Emitting signal for frame {self._frame_pos}')
            self.sigTrack.emit(self._frame, self._bboxes,
                               self._frame_pos)
            self._frame_pos = -1
            self._bboxes_pos = -1


