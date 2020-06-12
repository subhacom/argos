# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-01 11:50 PM
import logging
import threading
import numpy as np
from scipy import optimize
import cv2
from PyQt5 import (
    QtCore as qc,
    QtWidgets as qw)
from argos import utility as au

setup = au.init()

def xywh2xysr(x, y, w, h):
    return np.array((x + w / 2.0,
                     y + h / 2.0,
                     w * h,
                     w / float(h)))

def xysr2xywh(x, y, s, r):
    w = np.sqrt(s * r)
    h = s / w
    return np.array((x - w / 2.0,
                     y - h / 2.0,
                     w, h))


def pairwise_distance(new_bboxes, bboxes, boxtype, metric):
    """Takes two lists of boxes and computes the distance between every possible
     pair.

     new_bboxes: list of boxes as (x, y, w, h)

     bboxes: list of boxes as four x, y, w, h

     boxtype: OutlineStyle.bbox for axis aligned rectangle bounding box or
     OulineStyle.minrect for minimum area rotated rectangle

     metric: DistanceMetric, iou or euclidean. When euclidean, the squared
     Euclidean distance is used (calculating square root is expensive and
     unnecessary. If iou, use the area of intersection divided by the area
     of union.

     :returns `list` `[(ii, jj, dist), ...]` `dist` is the computed distance
     between `new_bboxes[ii]` and `bboxes[jj]`.

     """
    dist_list = []
    if metric == au.DistanceMetric.euclidean:
        centers = bboxes[:, :2] + bboxes[:, 2:] * 0.5
        new_centers = new_bboxes[:, :2] + new_bboxes[:, 2:] * 0.5
        for ii in range(len(new_bboxes)):
            for jj in range(len(bboxes)):
                dist = (new_centers[ii] - centers[jj]) ** 2
                dist_list.append((ii, jj, dist.sum()))
    elif metric == au.DistanceMetric.iou:
        if boxtype == au.OutlineStyle.bbox:  # This can be handled efficiently
            # Convert four anticlockwise vertices from top left into x, y, w, h
            for ii in range(len(new_bboxes)):
                for jj in range(len(bboxes)):
                    dist = 1.0 - au.rect_iou(bboxes[jj], new_bboxes[ii])
                    dist_list.append((ii, jj, dist))
        else:
            raise NotImplementedError('Only handling axis-aligned bounding boxes')
    else:
        raise NotImplementedError(f'Unknown metric {metric}')
    return dist_list


def match_bboxes(id_bboxes: dict, new_bboxes: np.ndarray,
                 boxtype: au.OutlineStyle,
                 metric: au.DistanceMetric = au.DistanceMetric.euclidean,
                 max_dist: int = 1e4
    ):
    """Match the bboxes in `new_bboxes` to the closest object.

    id_bboxes: dict mapping ids to bboxes

    new_bboxes: array of new bboxes to be matched to those in id_bboxes

    boxtype: OutlineStyle.bbox or OutlineStyle.minrect

    max_dist: anything that is more than this distance from all of the bboxes in
    id_bboxes are put in the unmatched list

    metric: iou for area of inetersection over union of the rectangles and
    euclidean for Euclidean distance between centers.

    :returns matched, new_unmatched, old_unmatched: `matched` is a dict mapping
    keys in `id_bboxes` to bbox indices in `new_bboxes` that are closest.
    `new_unmatched` is the set of indices into `bboxes` that did not match
    anything in `id_bboxes`, `old_unmatched` is the set of keys in `id_bboxes`
    whose corresponding bbox values did not match anything in `bboxes`.
    """
    logging.debug('Current bboxes: %r', id_bboxes)
    logging.debug('New bboxes: %r', new_bboxes)
    logging.debug('Box type: %r', boxtype)
    logging.debug('Max dist: %r', max_dist)
    if len(id_bboxes) == 0:
        return ({}, set(range(len(new_bboxes))), {})
    labels = list(id_bboxes.keys())
    bboxes = np.array(list(id_bboxes.values()), dtype=float)
    dist_list = pairwise_distance(new_bboxes, bboxes, boxtype=boxtype,
                                  metric=metric)
    dist_matrix = np.zeros((len(new_bboxes), len(id_bboxes)), dtype=float)
    for ii, jj, cost in dist_list:
        dist_matrix[ii, jj] = cost
    row_ind, col_ind = optimize.linear_sum_assignment(dist_matrix)
    matched = {}
    good_rows = set()
    good_cols = set()
    if metric == au.DistanceMetric.euclidean:
        max_dist *= max_dist
    for row, col in zip(row_ind, col_ind):
        if dist_matrix[row, col] < max_dist:
            good_rows.add(row)
            good_cols.add(labels[col])
            matched[labels[col]] = row
    new_unmatched = set(range(len(new_bboxes))) - good_rows
    old_unmatched = set(id_bboxes.keys()) - good_cols
    return matched, new_unmatched, old_unmatched



class KalmanTracker(object):
    """This class tries to improve performance over SORT or DeepSORT by using
    opencv's builtin Kalman Filter. OpenCV being written in C/C++ it outperforms
    the Python code in DeepSORT or filterpy (used in SORT).

    In my test, the predict step in OpenCV takes

    2.78 µs ± 14.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    compared to DeepSORT taking

    45.7 µs ± 1.24 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    """
    NDIM = 4
    DT = 1.0

    def __init__(self, bbox, track_id, n_init=3, max_age=10):
        super(KalmanTracker, self).__init__()
        self.state = au.TrackState.tentative
        self.hits = 1
        self.features = []
        self.time_since_update = 0
        self.n_init = n_init
        self.max_age = max_age

        self.filter = cv2.KalmanFilter(dynamParams=2 * self.NDIM,
                                       measureParams=self.NDIM, type=cv2.CV_64F)
        # Borrowing ideas from DeepSORT
        self.filter.measurementMatrix = np.array([
            [1., 0, 0, 0, 0, 0, 0, 0],
            [0, 1., 0, 0, 0, 0, 0, 0],
            [0, 0, 1., 0, 0, 0, 0, 0],
            [0, 0, 0, 1., 0, 0, 0, 0]
        ])
        # This is state transition matrix F
        self.filter.transitionMatrix = np.array([
            [1., 0, 0, 0, self.DT, 0, 0, 0],
            [0, 1., 0, 0, 0, self.DT, 0, 0],
            [0, 0, 1., 0, 0, 0, self.DT, 0],
            [0, 0, 0, 1., 0, 0, 0, self.DT],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        # NOTE state covariance matrix (P) is initialized as a function of
        # measured height in DeepSORT, but constant (as below) in SORT
        self.filter.errorCovPost = np.eye(2 * self.NDIM, dtype=float) * 10.0
        self.filter.errorCovPost[self.NDIM:, self.NDIM:] *= 1000.0  # High uncertainty for velocity at first
        # NOTE process noise covariance matrix (Q) [here motion covariance] is
        # computed as a function of mean height in DeepSORT, but constant
        # (as below) in SORT
        self.filter.processNoiseCov = np.eye(2 * self.NDIM, dtype=float)
        self.filter.processNoiseCov[self.NDIM:, self.NDIM:] *= 0.01
        self.filter.processNoiseCov[-1, -1] *= 0.01
        # Measurement noise covariance R
        self.filter.measurementNoiseCov = np.eye(self.NDIM, dtype=float)
        self.filter.measurementNoiseCov[2:, 2:] *= 10.0
        self.filter.statePost = np.r_[xywh2xysr(*bbox), np.zeros(self.NDIM)]

    @property
    def pos(self):
        return xysr2xywh(*self.filter.statePost[: self.NDIM])

    def predict(self):
        self.time_since_update += 1
        ret = self.filter.predict()
        return xysr2xywh(*ret[:self.NDIM].squeeze())

    def update(self, detection):
        pos = self.filter.correct(xywh2xysr(*detection))
        self.time_since_update = 0
        self.hits += 1
        if self.state == au.TrackState.tentative and self.hits >= self.n_init:
            self.state = au.TrackState.confirmed
        self.pos[:] = pos[:self.NDIM]
        return self.pos

    def mark_missed(self):
        if self.state == au.TrackState.tentative or \
                self.time_since_update > self.max_age:
            self.state = au.TrackState.deleted

    def is_deleted(self):
        return self.state == au.TrackState.deleted

    def is_confirmed(self):
        return self.state == au.TrackState.confirmed

    def is_tentative(self):
        return self.state == au.TrackState.tentative


class SORTracker(qc.QObject):
    """SORT algorithm implementation

    NOTE: accepts bounding boxes in (x, y, w, h) format.
    """
    sigTracked = qc.pyqtSignal(dict, int)

    def __init__(self, metric=au.DistanceMetric.iou, min_dist=0.3, max_age=1,
                 n_init=3, boxtype=au.OutlineStyle.bbox):
        super(SORTracker, self).__init__()
        self.n_init = n_init
        self.boxtype = boxtype
        self.metric = metric
        if self.metric == au.DistanceMetric.iou:
            self.min_dist = 1 - min_dist
        else:
            self.min_dist = min_dist
        self.max_age = max_age
        self.trackers = {}
        self._next_id = 1
        self._mutex = qc.QMutex()
        self._wait_cond = None

    @qc.pyqtSlot()
    def reset(self):
        self.trackers = {}
        self._next_id = 1

    @qc.pyqtSlot(threading.Event)
    def setWaitCond(self, cond: threading.Event) -> None:
        _ = qc.QMutexLocker(self._mutex)
        self._wait_cond = cond

    @qc.pyqtSlot(float)
    def setMinDist(self, dist: float) -> None:
        _ = qc.QMutexLocker(self._mutex)
        if self.metric == au.DistanceMetric.iou:
            self.min_dist = 1 - dist
        else:
            self.min_dist = dist

    @qc.pyqtSlot(int)
    def setMaxAge(self, max_age: int) -> None:
        """Set the maximum misses before discarding a track"""
        _ = qc.QMutexLocker(self._mutex)
        self.max_age = max_age

    @qc.pyqtSlot(int)
    def setMinHits(self, count: int) -> None:
        """Number of times a track should match prediction before it is
        confirmed"""
        _ = qc.QMutexLocker(self._mutex)
        self.n_init = count

    def update(self, bboxes):
        predicted_bboxes = {}
        for id_, tracker in self.trackers.items():
            prior = tracker.predict()
            if np.any(np.isnan(prior)) or np.any(prior[2:] < 0):
                continue
            predicted_bboxes[id_] = prior[:KalmanTracker.NDIM]
        self.trackers = {id_: self.trackers[id_] for id_ in predicted_bboxes}
        matched, new_unmatched, old_unmatched = match_bboxes(
            predicted_bboxes,
            bboxes[:, :KalmanTracker.NDIM],
            boxtype=self.boxtype,
            metric=self.metric,
            max_dist=self.min_dist)
        for track_id, bbox_id in matched.items():
            self.trackers[track_id].update(bboxes[bbox_id])
        for id_ in old_unmatched:
            self.trackers[id_].mark_missed()
        for ii in new_unmatched:
            self._add_tracker(bboxes[ii, :KalmanTracker.NDIM])
        self.trackers = {id_: tracker for id_, tracker in self.trackers.items()
                         if not tracker.is_deleted()}
        ret = {id_: tracker.pos for id_, tracker in
               self.trackers.items()}
        return ret

    def _add_tracker(self, bbox):
        self.trackers[self._next_id] = KalmanTracker(bbox, self._next_id,
                                                     self.n_init,
                                                     self.max_age)
        self._next_id += 1

    @qc.pyqtSlot(np.ndarray, int)
    def track(self, bboxes: np.ndarray, pos: int):
        _ = qc.QMutexLocker(self._mutex)
        if len(bboxes) == 0:
            ret = {}
        else:
            ret = self.update(bboxes)
        self.sigTracked.emit(ret, pos)
        if self._wait_cond is not None:
            logging.debug(f'Waiting on condition')
            self._wait_cond.wait()
        logging.debug(f'Finished frame {pos}')


class SORTWidget(qw.QWidget):
    sigTrack = qc.pyqtSignal(np.ndarray, int)
    sigTracked = qc.pyqtSignal(dict, int)
    sigQuit = qc.pyqtSignal()
    sigReset = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(SORTWidget, self).__init__(*args, **kwargs)
        self._max_age_label = qw.QLabel('Maximum age')
        self._max_age_label.setToolTip('Maximum number of misses before a track is removed')
        self._max_age_spin = qw.QSpinBox()
        self._max_age_spin.setRange(1, 100)
        self._max_age_spin.setValue(10)
        self._conf_age_label = qw.QLabel('Minimum hits')
        self._conf_age_label.setToolTip('Minimum number of hits before a track is confirmed')
        self._conf_age_spin = qw.QSpinBox()
        self._conf_age_spin.setRange(1, 100)
        self._conf_age_spin.setValue(3)
        self._min_dist_label = qw.QLabel('Minimum overlap')
        self._min_dist_spin = qw.QDoubleSpinBox()
        self._min_dist_spin.setRange(0.1, 1.0)
        self._min_dist_spin.setValue(0.3)
        self._disable_check = qw.QCheckBox('Disable tracking')
        layout = qw.QFormLayout()
        layout.addRow(self._min_dist_label, self._min_dist_spin)
        layout.addRow(self._conf_age_label, self._conf_age_spin)
        layout.addRow(self._max_age_label, self._max_age_spin)
        layout.addWidget(self._disable_check)
        self.tracker = SORTracker(metric=au.DistanceMetric.iou,
                                  min_dist=self._min_dist_spin.value(),
                                  max_age=self._max_age_spin.value(),
                                  n_init=self._conf_age_spin.value())
        self.thread = qc.QThread()
        self.tracker.moveToThread(self.thread)
        self._max_age_spin.valueChanged.connect(self.tracker.setMaxAge)
        self._min_dist_spin.valueChanged.connect(self.tracker.setMinDist)
        self._conf_age_spin.valueChanged.connect(self.tracker.setMinHits)
        self._disable_check.stateChanged.connect(self.disable)
        self.sigTrack.connect(self.tracker.track)
        self.tracker.sigTracked.connect(self.sigTracked)
        self.sigReset.connect(self.tracker.reset)
        self.sigQuit.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.setLayout(layout)
        self.thread.start()

    @qc.pyqtSlot(int)
    def disable(self, state):
        self.sigTrack.disconnect()
        if state:
            self.sigTrack.connect(self.sendDummySigTracked)
        else:
            self.sigTracked.connect(self.tracker.track)

    @qc.pyqtSlot(np.ndarray, int)
    def sendDummySigTracked(self, bboxes: np.ndarray, pos: int) -> None:
        self.sigTracked.emit({ii: bboxes[ii] for ii in range(bboxes.shape[0])},
                             pos)



