# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-01 11:50 PM

"""
===========================================
SORT algorithm for multiple object tracking
===========================================

Implemented with OpenCV's builtin Kalman filter.

This algorithm was originally developed and published by Alex Bewly [2]_.

Original implementation of SORT_ and DeepSORT_

References
----------
.. [2] Bewley, A., Ge, Z., Ott, L., Ramos, F. & Upcroft, B. Simple
   Online and Realtime Tracking. 2016 IEEE International Conference on
   Image Processing (ICIP) 34643468 (2016)
   doi:10.1109/ICIP.2016.7533003.

.. _SORT: https://github.com/abewley/sort

.. _DeepSORT: https://github.com/nwojke/deep_sort.

"""
import logging
import numpy as np
import cv2
from PyQt5 import (
    QtCore as qc)

import argos.constants
from argos import utility as au
from argos.utility import match_bboxes

settings = au.init()


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
            self.filter.errorCovPost[self.NDIM:,
            self.NDIM:] *= 1000.0  # High uncertainty for velocity at first

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


class SORTracker(object):
    """SORT algorithm implementation

    NOTE: accepts bounding boxes in (x, y, w, h) format.
    """
    sigTracked = qc.pyqtSignal(dict, int)

    def __init__(self, metric=argos.constants.DistanceMetric.iou, min_dist=0.3,
                 max_age=1,
                 n_init=3, min_hits=3,
                 boxtype=argos.constants.OutlineStyle.bbox):
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

    def reset(self):
        logging.debug('Resetting trackers.')
        self.trackers = {}
        self._next_id = 1
        self.frame_count = 0

    def setDistMetric(self, metric: argos.constants.DistanceMetric) -> None:
        if self.metric != metric:
            self.metric = metric
            if self.min_dist < 1:
                self.min_dist = 1 - self.min_dist

    def setMinDist(self, dist: float) -> None:
        if self.metric == argos.constants.DistanceMetric.iou:
            self.min_dist = 1 - dist
        else:
            self.min_dist = dist

    def update(self, bboxes):
        """Update the trackers with new bboxes.

        Parameters
        ----------
        bboxes: np.ndarray
            a 2D array with one row for each bbox in (x, y, w, h) format.

        Returns
        -------
        dict
            dict mapping object ids to tracked position
        """
        predicted_bboxes = {}
        for id_, tracker in self.trackers.items():
            prior = tracker.predict()
            if np.any(np.isnan(prior)) or np.any(
                    prior[:KalmanTracker.NDIM] < 0):
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


