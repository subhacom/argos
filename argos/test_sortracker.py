# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-08 1:36 PM
import sys
import logging
import threading
import numpy as np
from scipy import optimize
import cv2
from typing import Dict, Tuple, OrderedDict, List, Set

import argos.constants
from argos import utility as au


logging.getLogger().addHandler(logging.FileHandler('log_argos_sort.log', mode='w'))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger().setLevel(logging.DEBUG)

def xywh2xysr(x, y, w, h):
    return np.array((x + w / 2.0,
                     y + h / 2.0,
                     w * h, #w / float(h), #
                     w / float(h))) # h)) #

def xysr2xywh(x, y, s, r):
    w = np.sqrt(s * r)
    h = s / w
    # w = s * r
    # h = r
    return np.array((x - w / 2.0,
                     y - h / 2.0,
                     w, h))


def pairwise_distance(new_bboxes: np.ndarray, bboxes: np.ndarray,
                      boxtype: argos.utility.OutlineStyle,
                      metric: argos.utility.DistanceMetric) -> np.ndarray:
    """Takes two lists of boxes and computes the distance between every possible
    pair.

    Parameters
    ----------
    new_bboxes: np.ndarray
       Array of bounding boxes, each row as (x, y, w, h)
    bboxes: np.ndarray
       Array of bounding boxes, each row as (x, y, w, h)
    boxtype: OutlineStyle
       OutlineStyle.bbox for axis aligned rectangle bounding box or
       OulineStyle.minrect for minimum area rotated rectangle
    metric: DistanceMetric
       iou or euclidean. When euclidean, the squared Euclidean distance is
       used (calculating square root is expensive and unnecessary. If iou, use
       the area of intersection divided by the area of union.
    Returns
    --------
    np.ndarray
        row ``ii``, column ``jj`` contains the computed distance `between
        ``new_bboxes[ii]`` and ``bboxes[jj]``.
     """
    dist = np.zeros((new_bboxes.shape[0], bboxes.shape[0]), dtype=np.float)
    if metric == argos.utility.DistanceMetric.euclidean:
        centers = bboxes[:, :2] + bboxes[:, 2:] * 0.5
        new_centers = new_bboxes[:, :2] + new_bboxes[:, 2:] * 0.5
        for ii in range(len(new_bboxes)):
            for jj in range(len(bboxes)):
                dist[ii, jj] = np.sum((new_centers[ii] - centers[jj]) ** 2)
    elif metric == argos.utility.DistanceMetric.iou:
        if boxtype == argos.utility.OutlineStyle.bbox:  # This can be handled efficiently
            for ii in range(len(new_bboxes)):
                for jj in range(len(bboxes)):
                    dist[ii, jj] = 1.0 - argos.utility.rect_iou(bboxes[jj],
                                                  new_bboxes[ii])
        else:
            raise NotImplementedError(
                'Only handling axis-aligned bounding boxes')
    else:
        raise NotImplementedError(f'Unknown metric {metric}')
    return dist


def match_bboxes(id_bboxes: dict, new_bboxes: np.ndarray,
                 boxtype: argos.utility.OutlineStyle,
                 metric: argos.utility.DistanceMetric = argos.utility.DistanceMetric.euclidean,
                 max_dist: float = 10000
                 ) -> Tuple[Dict[int, int], Set[int], Set[int]]:
    """Match the bboxes in `new_bboxes` to the closest object in the
    ``id_bboxes`` dictionary.

    Parameters
    ----------
    id_bboxes: dict[int, np.ndarray]
        Mapping ids to bounding boxes
    new_bboxes: np.ndarray
        Array of new bounding boxes to be matched to those in ``id_bboxes``.
    boxtype: {OutlineStyle.bbox, OutlineStyle.minrect}
        Type of bounding box to match.
    max_dist: int
        Anything that is more than this distance from all of the bboxes in
        ``id_bboxes`` are put in the unmatched list
    metric: {DistanceMetric.euclidean, DistanceMetric.iou}
        iou for area of inetersection over union of the rectangles,
        and euclidean for Euclidean distance between centers.

    Returns
    -------
    matched: dict[int, int]
        Mapping keys in ``id_bboxes`` to bbox indices in ``new_bboxes`` that are
        closest.
    new_unmatched: set[int]
        Set of indices into `bboxes` that did not match anything in
        ``id_bboxes``
    old_unmatched: set[int]
        Set of keys in ``id_bboxes`` whose corresponding bbox values did not
        match anything in ``bboxes``.
    """
    logging.debug('Current bboxes:\n%r', np.array(list(id_bboxes.items())))
    logging.debug('New bboxes:\n%r', np.array(new_bboxes))
    logging.debug('Box type: %r', boxtype)
    logging.debug('Max dist: %r', max_dist)
    if len(id_bboxes) == 0:
        return ({}, set(range(len(new_bboxes))), {})
    labels = list(id_bboxes.keys())
    bboxes = np.array(list(id_bboxes.values()), dtype=float)
    dist_matrix = pairwise_distance(new_bboxes, bboxes, boxtype=boxtype,
                                    metric=metric)
    logging.debug(f'Distance matrix\n{dist_matrix}')
    row_ind, col_ind = optimize.linear_sum_assignment(dist_matrix)
    matched = {}
    good_rows = set()
    good_cols = set()
    if metric == argos.utility.DistanceMetric.euclidean:
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
        self.filter = cv2.KalmanFilter(dynamParams=2 * self.NDIM - 1,
                                       measureParams=self.NDIM, type=cv2.CV_64F)
        # Borrowing ideas from SORT/DeepSORT
        # Measurement marix H
        self.filter.measurementMatrix = np.array([
            [1., 0, 0, 0, 0, 0, 0],
            [0, 1., 0, 0, 0, 0, 0],
            [0, 0, 1., 0, 0, 0, 0],
            [0, 0, 0, 1., 0, 0, 0]
        ])
        # This is state transition matrix F
        self.filter.transitionMatrix = np.array([
            [1., 0, 0, 0, self.DT, 0, 0],
            [0, 1., 0, 0, 0, self.DT, 0],
            [0, 0, 1., 0, 0, 0, self.DT],
            [0, 0, 0, 1., 0, 0, 0],
            [0, 0, 0, 0, self.DT, 0, 0],
            [0, 0, 0, 0, 0, self.DT, 0],
            [0, 0, 0, 0, 0, 0, self.DT],
        ])
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
            self.filter.errorCovPost = np.eye(2 * self.NDIM-1, dtype=float) * 10.0
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
            self.filter.processNoiseCov = np.eye(2 * self.NDIM-1)
            # self.filter.processNoiseCov[2, 2] = 1e-2
            self.filter.processNoiseCov[self.NDIM:, self.NDIM:] *= 0.01
            self.filter.processNoiseCov[-1, -1] *= 0.01
            # ~~~~ Till here is according to SORT

        # Measurement noise covariance R
        if not self.cov_deepsort:
            # ~~~~ This is according to SORT
            self.filter.measurementNoiseCov = np.eye(self.NDIM)
            self.filter.measurementNoiseCov[2:, 2:] *= 10.0
            # ~~~~ Till here is according to SORT
        self.filter.statePost = np.r_[xywh2xysr(*bbox), np.zeros(self.NDIM-1)]
        # logging.info(f'kf.P:\n{self.filter.errorCovPost}')
        # logging.info(f'kf.Q:\n{self.filter.processNoiseCov}')
        # logging.info(f'kf.R:\n{self.filter.measurementNoiseCov}')

    @property
    def pos(self):
        return xysr2xywh(*self.filter.statePost[: self.NDIM])

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
        # logging.info(f'====== {self.tid}    Prediction')
        # logging.info(f'------ x:\n{self.filter.statePost}')
        # logging.info(f'------ F:\n{self.filter.transitionMatrix}')
        # logging.info(f'------ P:\n{self.filter.errorCovPost}')
        # logging.info(f'------ Q:\n{self.filter.processNoiseCov}')
        ret = self.filter.predict()
        # logging.info(f'------ x_prior:\n{self.filter.statePre}')
        # logging.info(f'------ P_prior:\n{self.filter.errorCovPre}')
        # logging.info(f'****** {self.tid} Predicted\n{ret}')
        return xysr2xywh(*ret[:self.NDIM].squeeze())

    def update(self, detection):
        if self.cov_deepsort:
            # ~~ This follows deepSORT
            measure_cov = [self._std_weight_pos * self.filter.statePost[3],
                           self._std_weight_pos * self.filter.statePost[3],
                           1e-1,
                           self._std_weight_pos * self.filter.statePost[3]]
            self.filter.measurementNoiseCov = np.diag(np.square(measure_cov))
            # ~~ till here follows deepSORT
        # logging.info(f'======== {self.tid} == Correction')
        # logging.info(f'-------- z:\n{detection}')
        # logging.info(f'------ H:\n{self.filter.measurementMatrix}')
        # logging.info(f'------ P_prior:\n{self.filter.errorCovPre}')
        # logging.info(f'------ x_prior:\n{self.filter.statePre}')
        # logging.info(f'------ R:\n{self.filter.measurementNoiseCov}')
        pos = self.filter.correct(xywh2xysr(*detection))
        # logging.info(f'------ K:\n{self.filter.gain}')
        # logging.info(f'------ P:\n{self.filter.errorCovPost}')
        # logging.info(f'------ x:\n{self.filter.statePost}')

        self.time_since_update = 0
        self.hits += 1
        self.pos[:] = pos[:self.NDIM]
        return self.pos


class SORTracker(object):
    """SORT algorithm implementation

    NOTE: accepts bounding boxes in (x, y, w, h) format.
    """
    def __init__(self, metric=argos.constants.DistanceMetric.iou, min_dist=0.3, max_age=1,
                 n_init=3, min_hits=3, boxtype=argos.constants.OutlineStyle.bbox):
        super(SORTracker, self).__init__()
        self.n_init = n_init
        self.min_hits = min_hits
        self.boxtype = boxtype
        self.metric = metric
        if metric == argos.constants.DistanceMetric.iou:
            self.min_dist = 1 - min_dist
        else:
            self.min_dist = min_dist
        self.max_age = max_age
        self.trackers = {}
        self._next_id = 1
        self.frame_count = 0

    def reset(self):
        self.trackers = {}
        self._next_id = 1

    # @qc.pyqtSlot(float)
    def setMinDist(self, dist: float) -> None:
        # _ = qc.QMutexLocker(self._mutex)
        if self.metric == argos.constants.DistanceMetric.iou:
            self.min_dist = 1 - dist
        else:
            self.min_dist = dist

    # @qc.pyqtSlot(int)
    def setMaxAge(self, max_age: int) -> None:
        """Set the maximum misses before discarding a track"""
        # _ = qc.QMutexLocker(self._mutex)
        self.max_age = max_age

   # @qc.pyqtSlot(int)
    def setMinHits(self, count: int) -> None:
        """Number of times a track should match prediction before it is
        confirmed"""
        # _ = qc.QMutexLocker(self._mutex)
        self.n_init = count

    def update(self, bboxes):
        self.frame_count += 1
        predicted_bboxes = {}
        for id_, tracker in self.trackers.items():
            prior = tracker.predict()
            if np.any(np.isnan(prior)):
                logging.info(f'Found nan in prior of {id_}')
                continue
            predicted_bboxes[id_] = prior
        logging.info('********* predicted')
        logging.info(np.array(list(predicted_bboxes.items())))
        logging.info('------')
        self.trackers = {id_: self.trackers[id_] for id_ in predicted_bboxes}
        matched, new_unmatched, old_unmatched = match_bboxes(
            predicted_bboxes,
            bboxes[:, :KalmanTracker.NDIM],
            boxtype=self.boxtype,
            metric=self.metric,
            max_dist=self.min_dist)
        logging.info(f'--- Matches\n{np.array(list(matched.items()))}')
        logging.info(f'---- Unmatched new\n{new_unmatched}')
        logging.info(f'-----Unmatched old\n{old_unmatched}')
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
        logging.info(
            f'After deleting dead trackers\n{np.array(list(self.trackers.items()))}')
        return ret

    def _add_tracker(self, bbox):
        self.trackers[self._next_id] = KalmanTracker(bbox, self._next_id,
                                                     self.n_init,
                                                     self.max_age)
        self._next_id += 1


import pandas as pd

def test():
    dfile = 'C:/Users/raysu/Documents/src/argos/test_data/2020_02_20_00267_test.avi.h5'
    vfile = 'C:/Users/raysu/Documents/src/argos/test_data/2020_02_20_00267_test.avi'
    store = pd.HDFStore(dfile)
    detections = store['segmented']
    video = cv2.VideoCapture(vfile)
    win = 'Argos'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    tracker = SORTracker(max_age=10, n_init=3, min_dist=0.3)
    tracked = []
    for frameno, dgrp in detections.groupby('frame'):
        v = dgrp.values[:, 1:].copy()
        logging.info(f'FFFFFF Frame {frameno} ===')
        logging.info(f'{v}\n~~~~~\n')
        if frameno % 100 == 0:
            logging.info(f'Frame {frameno} processed')
        # v = np.c_[v, np.ones(v.shape[0], dtype=int)]
        track_bbs_id = tracker.update(v)
        video.set(cv2.CAP_PROP_POS_FRAMES, frameno)
        ret, frame = video.read()
        tracked.append(track_bbs_id)
        for bbs_id, bbox in track_bbs_id.items():
            logging.info(f'--- {bbs_id} ---')
            bbox = np.int0(bbox)
            logging.info(bbox)
            cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[:2] + bbox[2:4]), 255)
            cv2.putText(frame, str(frameno), (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
            cv2.putText(frame, str(bbs_id), tuple(bbox[:2]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
        cv2.imshow(win, frame)
        key = cv2.waitKey(1000)
        if key == 27 or key == ord('q'):
            break

    # tracked = np.concatenate(tracked)

if __name__ == '__main__':
    test()
