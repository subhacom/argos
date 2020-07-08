# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-01 11:50 PM
import logging
import threading
import numpy as np
import cv2
from PyQt5 import (
    QtCore as qc,
    QtWidgets as qw)

import argos.constants
from argos import utility as au
from argos.utility import match_bboxes

setup = au.init()


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

    def __init__(self, bbox, track_id, min_hits=3, max_age=10, deepsort=False):
        """bbox is in xywh format and converted to xyrh format"""
        super(KalmanTracker, self).__init__()
        self.tid = track_id
        self.hits = 1
        self.min_hits = min_hits
        self.features = []
        self.time_since_update = 0
        self.n_init = min_hits
        self.max_age = max_age
        self._std_weight_pos = 1.0 / 20
        self._std_weight_vel = 1.0 / 160
        # flag to switch between fixed covariances like SORT vs
        # measurement-based covariance like DeepSORT
        self.cov_deepsort = deepsort
        self.filter = cv2.KalmanFilter(dynamParams=2 * self.NDIM,
                                       measureParams=self.NDIM, type=cv2.CV_64F)
        # Borrowing ideas from SORT/DeepSORT
        # Measurement marix H
        self.filter.measurementMatrix = np.eye(self.NDIM, 2 * self.NDIM)

        # This is state transition matrix F
        self.filter.transitionMatrix = np.eye(2 * self.NDIM, 2 * self.NDIM)
        for ii in range(self.NDIM):
            self.filter.transitionMatrix[ii, ii + self.NDIM] = self.DT
        # NOTE state covariance matrix (P) is initialized as a function of
        # measured height in DeepSORT, but constant in SORT.
        if self.cov_deepsort:
            error_cov = [2 * self._std_weight_pos * bbox[3],
                         2 * self._std_weight_pos * bbox[3],
                         1e-2,
                         2 * self._std_weight_pos * bbox[3],
                         10 * self._std_weight_vel * bbox[3],
                         10 * self._std_weight_vel * bbox[3],
                         1e-5,
                         10 * self._std_weight_vel * bbox[3]]
            self.filter.errorCovPost = np.diag(np.square(error_cov))
        else:
            self.filter.errorCovPost = np.eye(2 * self.NDIM, dtype=float) * 10.0
            self.filter.errorCovPost[self.NDIM:, self.NDIM:] *= 1000.0  # High uncertainty for velocity at first

        # NOTE process noise covariance matrix (Q) [here motion covariance] is
        # computed as a function of mean height in DeepSORT, but constant
        # in SORT

        if self.cov_deepsort:
            proc_cov = [self._std_weight_pos * bbox[3],
                        self._std_weight_pos * bbox[3],
                        1e-2,
                        self._std_weight_pos * bbox[3],
                        self._std_weight_vel * bbox[3],
                        self._std_weight_vel * bbox[3],
                        1e-5,
                        self._std_weight_vel * bbox[3]]
            self.filter.processNoiseCov = np.diag(np.square(proc_cov))
            # ~~ till here follows deepSORT
        else:
            # ~~~~ This is according to SORT
            self.filter.processNoiseCov = np.eye(2 * self.NDIM)
            # self.filter.processNoiseCov[2, 2] = 1e-2
            self.filter.processNoiseCov[self.NDIM:, self.NDIM:] *= 0.01
            self.filter.processNoiseCov[-2:, -2:] *= 0.01
            # ~~~~ Till here is according to SORT

        # Measurement noise covariance R
        if not self.cov_deepsort:
            # ~~~~ This is according to SORT
            self.filter.measurementNoiseCov = np.eye(self.NDIM)
            self.filter.measurementNoiseCov[2:, 2:] *= 10.0
            # ~~~~ Till here is according to SORT
        self.filter.statePost = np.r_[au.tlwh2xyrh(bbox), np.zeros(self.NDIM)]

    @property
    def pos(self):
        return au.xyrh2tlwh(self.filter.statePost[: self.NDIM])

    def predict(self):
        if self.cov_deepsort:
            # ~~ This follows deepSORT
            proc_cov = [self._std_weight_pos * self.filter.statePost[3],
                        self._std_weight_pos * self.filter.statePost[3],
                        1e-2,
                        self._std_weight_pos * self.filter.statePost[3],
                        self._std_weight_vel * self.filter.statePost[3],
                        self._std_weight_vel * self.filter.statePost[3],
                        1e-5,
                        self._std_weight_vel * self.filter.statePost[3]]
            self.filter.processNoiseCov = np.diag(np.square(proc_cov))
            # ~~ till here follows deepSORT
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1
        ret = self.filter.predict()
        return au.xyrh2tlwh(ret[:self.NDIM].squeeze())

    def update(self, detection):
        if self.cov_deepsort:
            # ~~ This follows deepSORT
            measure_cov = [self._std_weight_pos * self.filter.statePost[3],
                           self._std_weight_pos * self.filter.statePost[3],
                           1e-1,
                           self._std_weight_pos * self.filter.statePost[3]]
            self.filter.measurementNoiseCov = np.diag(np.square(measure_cov))
            # ~~ till here follows deepSORT
        pos = self.filter.correct(au.tlwh2xyrh(detection))
        self.time_since_update = 0
        self.hits += 1
        self.pos[:] = pos[:self.NDIM]
        return self.pos



class SORTracker(qc.QObject):
    """SORT algorithm implementation

    NOTE: accepts bounding boxes in (x, y, w, h) format.
    """
    sigTracked = qc.pyqtSignal(dict, int)

    def __init__(self, metric=argos.constants.DistanceMetric.iou, min_dist=0.3, max_age=1,
                 n_init=3, min_hits=3, boxtype=argos.constants.OutlineStyle.bbox):
        super(SORTracker, self).__init__()
        self.n_init = n_init
        self.min_hits = min_hits
        self.boxtype = boxtype
        self.metric = metric
        if self.metric == argos.constants.DistanceMetric.iou:
            self.min_dist = 1 - min_dist
        else:
            self.min_dist = min_dist
        self.max_age = max_age
        self.trackers = {}
        self._next_id = 1
        self.frame_count = 0
        self._mutex = qc.QMutex()
        self._wait_cond = None

    @qc.pyqtSlot()
    def reset(self):
        logging.debug('Resetting trackers.')
        self.trackers = {}
        self._next_id = 1
        self.frame_count = 0

    @qc.pyqtSlot(threading.Event)
    def setWaitCond(self, cond: threading.Event) -> None:
        _ = qc.QMutexLocker(self._mutex)
        self._wait_cond = cond

    @qc.pyqtSlot(float)
    def setMinDist(self, dist: float) -> None:
        _ = qc.QMutexLocker(self._mutex)
        if self.metric == argos.constants.DistanceMetric.iou:
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
            if np.any(np.isnan(prior)) or np.any(prior[:KalmanTracker.NDIM] < 0):
                logging.info(f'Found nan or negative in prior of {id_}')
                continue
            predicted_bboxes[id_] = prior[:KalmanTracker.NDIM]
        self.trackers = {id_: self.trackers[id_] for id_ in predicted_bboxes}
        for id_, bbox in predicted_bboxes.items():
            if np.any(bbox < 0):
                logging.debug(f'EEEE prediced bbox negative: {id_}: {bbox}')
        matched, new_unmatched, old_unmatched = match_bboxes(
            predicted_bboxes,
            bboxes[:, :KalmanTracker.NDIM],
            boxtype=self.boxtype,
            metric=self.metric,
            max_dist=self.min_dist)
        for track_id, bbox_id in matched.items():
            self.trackers[track_id].update(bboxes[bbox_id])
        for ii in new_unmatched:
            self._add_tracker(bboxes[ii, :KalmanTracker.NDIM])
        ret = {}
        for id_ in list(self.trackers.keys()):
            tracker = self.trackers[id_]
            if (tracker.time_since_update < 1) and \
                (tracker.hits >= self.min_hits or
                 self.frame_count <= self.min_hits):
                ret[id_] = tracker.pos
            if tracker.time_since_update > self.max_age:
                self.trackers.pop(id_)
        return ret

    def _add_tracker(self, bbox):
        self.trackers[self._next_id] = KalmanTracker(bbox, self._next_id,
                                                     self.n_init,
                                                     self.max_age)
        self._next_id += 1

    @qc.pyqtSlot(dict, int)
    def track(self, bboxes: dict, pos: int) -> None:
        logging.debug(f'Received from {self.sender()} bboxes: {bboxes}')
        _ = qc.QMutexLocker(self._mutex)
        if len(bboxes) == 0:
            ret = {}
        else:
            ret = self.update(bboxes)
        logging.debug(f'SORTracker: frame {pos}, Rectangles: \n{ret}')
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
        self._max_age_spin.setToolTip(self._max_age_label.toolTip())
        self._conf_age_label = qw.QLabel('Minimum hits')
        self._conf_age_label.setToolTip('Minimum number of hits before a track is confirmed')
        self._conf_age_spin = qw.QSpinBox()
        self._conf_age_spin.setRange(1, 100)
        self._conf_age_spin.setValue(3)
        self._conf_age_spin.setToolTip(self._conf_age_label.toolTip())
        self._min_dist_label = qw.QLabel('Minimum overlap')
        self._min_dist_spin = qw.QDoubleSpinBox()
        self._min_dist_spin.setRange(0.1, 1.0)
        self._min_dist_spin.setValue(0.3)
        self._min_dist_spin.setToolTip('Minimum overlap between bounding boxes '
                                       'to consider them same object.')
        self._disable_check = qw.QCheckBox('Disable tracking')
        self._disable_check.setToolTip('Just show the identified objects. Can '
                                       'be useful for troubleshooting.')
        layout = qw.QFormLayout()
        self.setLayout(layout)
        layout.addRow(self._min_dist_label, self._min_dist_spin)
        layout.addRow(self._conf_age_label, self._conf_age_spin)
        layout.addRow(self._max_age_label, self._max_age_spin)
        layout.addWidget(self._disable_check)
        self.tracker = SORTracker(metric=argos.constants.DistanceMetric.iou,
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
        self.thread.start()

    @qc.pyqtSlot(int)
    def disable(self, state):
        self.sigTrack.disconnect()
        if state:
            self.sigTrack.connect(self.sendDummySigTracked)
        else:
            self.sigTrack.connect(self.tracker.track)

    @qc.pyqtSlot(np.ndarray, int)
    def sendDummySigTracked(self, bboxes: np.ndarray, pos: int) -> None:
        ret = {ii+1: bboxes[ii] for ii in range(bboxes.shape[0])}
        self.sigTracked.emit(ret, pos)

    @qc.pyqtSlot(np.ndarray, int)
    def track(self, bboxes: np.ndarray, pos: int) -> None:
        """Just to intercept signal source for debugging"""
        logging.debug(f'Received frame {pos} from {self.sender()} bboxes: {bboxes}')
        self.sigTrack.emit(bboxes, pos)



def test():
    import sys
    app = qw.QApplication(sys.argv)
    win = SORTWidget()
    win.setMinimumSize(800, 600)
    # win.setWindowTitle('Argos - track animals in video')
    # win.showMaximized()
    # app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    test()