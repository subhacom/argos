# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
"""
================================================
ByteTrack-style motion-only multi-object tracker
================================================

Two-stage IoU matching with Tentative / Confirmed / Lost track states.
No appearance or re-ID features required — suited for animals that are
not individually identifiable (rats, grasshoppers, etc.).

Key difference from SORT:
- SORT discards a track the moment a detection is missing for one frame.
- ByteTracker keeps it in a *Lost* pool for up to ``max_age`` frames and
  attempts re-association with every new set of detections before
  finally ageing it out.

Reference
---------
Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box", ECCV 2022.  (The score-splitting step is omitted here
because detections have already been filtered upstream.)
"""
import logging
import numpy as np

import argos.constants
from argos import utility as au
from argos.utility import match_bboxes
from argos.sortracker import KalmanTracker

settings = au.init()

_TENTATIVE = 0
_CONFIRMED = 1
_LOST = 2


class ByteTracker:
    """Motion-only tracker using two-stage IoU matching.

    Parameters
    ----------
    min_hits : int
        Consecutive detections required to confirm a new track.
    max_age : int
        Maximum frames a lost track is kept before deletion.
    iou_threshold : float
        Minimum IoU between predicted and detected bbox to accept a match.
    boxtype : OutlineStyle
        Bounding-box representation (only bbox supported).
    """

    def __init__(
        self,
        min_hits: int = 3,
        max_age: int = 30,
        iou_threshold: float = 0.3,
        boxtype: argos.constants.OutlineStyle = argos.constants.OutlineStyle.bbox,
    ):
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.boxtype = boxtype
        self._tracks: dict = {}   # track_id → KalmanTracker
        self._states: dict = {}   # track_id → _TENTATIVE | _CONFIRMED | _LOST
        self._next_id = 1
        self.frame_count = 0

    def reset(self) -> None:
        logging.debug('Resetting ByteTracker.')
        self._tracks.clear()
        self._states.clear()
        self._next_id = 1
        self.frame_count = 0

    def update(self, bboxes: np.ndarray) -> dict:
        """Update tracker with new detections.

        Parameters
        ----------
        bboxes : np.ndarray
            Shape (N, 4) array of detections in (x, y, w, h) format,
            dtype int64.

        Returns
        -------
        dict
            ``{track_id: bbox}`` for every confirmed track actively
            matched this frame.
        """
        self.frame_count += 1
        # IoU distance threshold: lower distance = higher overlap
        max_dist = 1.0 - self.iou_threshold

        # ── Predict next positions ────────────────────────────────────────
        predicted: dict = {}
        for tid in list(self._tracks):
            prior = self._tracks[tid].predict()
            if not (np.any(np.isnan(prior)) or
                    np.any(prior[:KalmanTracker.NDIM] < 0)):
                predicted[tid] = prior[:KalmanTracker.NDIM]
            else:
                del self._tracks[tid]
                del self._states[tid]

        active_ids = [tid for tid in predicted
                      if self._states[tid] in (_TENTATIVE, _CONFIRMED)]
        lost_ids = [tid for tid in predicted
                    if self._states[tid] == _LOST]

        # ── Stage 1: match all detections → active tracks ─────────────────
        if len(bboxes) > 0 and len(active_ids) > 0:
            active_pred = {tid: predicted[tid] for tid in active_ids}
            matched1, unmatched_det_idx, unmatched_active = match_bboxes(
                active_pred,
                bboxes[:, :KalmanTracker.NDIM],
                boxtype=self.boxtype,
                metric=argos.constants.DistanceMetric.iou,
                max_dist=max_dist,
            )
        else:
            matched1 = {}
            unmatched_det_idx = set(range(len(bboxes)))
            unmatched_active = set(active_ids)

        # Update matched active tracks
        for tid, det_i in matched1.items():
            self._tracks[tid].update(bboxes[det_i])
            if (self._states[tid] == _TENTATIVE and
                    self._tracks[tid].hits >= self.min_hits):
                self._states[tid] = _CONFIRMED

        # Unmatched active: tentative → delete, confirmed → lost
        for tid in list(unmatched_active):
            if self._states[tid] == _TENTATIVE:
                del self._tracks[tid]
                del self._states[tid]
            else:
                self._states[tid] = _LOST
                self._tracks[tid].time_since_update = 1

        # ── Stage 2: match remaining detections → lost tracks ─────────────
        rem_list = sorted(unmatched_det_idx)
        if len(rem_list) > 0 and len(lost_ids) > 0:
            rem_bboxes = bboxes[rem_list]
            lost_pred = {tid: predicted[tid] for tid in lost_ids}
            matched2, unmatched_det2, unmatched_lost = match_bboxes(
                lost_pred,
                rem_bboxes[:, :KalmanTracker.NDIM],
                boxtype=self.boxtype,
                metric=argos.constants.DistanceMetric.iou,
                max_dist=max_dist,
            )
            # Re-associate matched lost tracks
            for tid, rem_i in matched2.items():
                orig_i = rem_list[rem_i]
                self._tracks[tid].update(bboxes[orig_i])
                self._states[tid] = _CONFIRMED
                self._tracks[tid].time_since_update = 0
            # Age unmatched lost tracks
            for tid in unmatched_lost:
                if tid in self._tracks:
                    self._tracks[tid].time_since_update += 1
                    if self._tracks[tid].time_since_update > self.max_age:
                        del self._tracks[tid]
                        del self._states[tid]
            new_det_indices = [rem_list[i] for i in unmatched_det2]
        else:
            # No lost tracks to match against — just age them
            for tid in lost_ids:
                if tid in self._tracks:
                    self._tracks[tid].time_since_update += 1
                    if self._tracks[tid].time_since_update > self.max_age:
                        del self._tracks[tid]
                        del self._states[tid]
            new_det_indices = rem_list

        # ── Create tentative tracks for still-unmatched detections ─────────
        for det_i in new_det_indices:
            self._add_track(bboxes[det_i, :KalmanTracker.NDIM])

        # ── Return confirmed tracks matched this frame ─────────────────────
        result: dict = {}
        for tid, tracker in self._tracks.items():
            if (self._states.get(tid) == _CONFIRMED
                    and tracker.time_since_update == 0
                    and (tracker.hits >= self.min_hits
                         or self.frame_count <= self.min_hits)):
                result[tid] = tracker.pos
        return result

    def _add_track(self, bbox: np.ndarray) -> None:
        self._tracks[self._next_id] = KalmanTracker(
            bbox, self._next_id, self.min_hits, self.max_age
        )
        self._states[self._next_id] = _TENTATIVE
        self._next_id += 1
