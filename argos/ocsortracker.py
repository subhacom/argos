# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
"""
==============================================
OC-SORT: Observation-Centric SORT tracker
==============================================

Three improvements over SORT, none requiring appearance/re-ID features:

1. **OCM** (Observation-Centric Momentum): adds a velocity-direction
   consistency term to the IoU cost matrix — penalises matches that
   require the animal to reverse direction.

2. **ORU** (Observation-Centric Re-Update): when a lost track is
   re-associated after a gap of K frames, the Kalman filter state is
   restored to the last-seen snapshot and replayed through K-1 linearly-
   interpolated virtual observations so that velocity/covariance are
   smoothly corrected rather than left in an occluded-drift state.

3. **OCR** (Observation-Centric Recovery): the second matching stage
   uses each track's *last real observation* position (rather than the
   drifted Kalman prediction) to match remaining detections.

Reference
---------
Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust
Multi-Object Tracking", CVPR 2023.  https://arxiv.org/abs/2203.14360
"""
import logging
import numpy as np
from scipy.optimize import linear_sum_assignment

import argos.constants
from argos import utility as au
from argos.utility import pairwise_distance
from argos.sortracker import KalmanTracker

settings = au.init()

_TENTATIVE = 0
_CONFIRMED = 1
_LOST = 2


class _OCSortTrack:
    """KalmanTracker augmented with OC-SORT observation history.

    Wraps a :class:`~argos.sortracker.KalmanTracker` and adds:
    - observation history (last ``delta_t + 2`` frames) for velocity
    - Kalman posterior snapshot for ORU state restoration
    """

    def __init__(
        self,
        bbox: np.ndarray,
        track_id: int,
        min_hits: int,
        max_age: int,
        delta_t: int = 3,
    ) -> None:
        self._kt = KalmanTracker(bbox, track_id, min_hits, max_age)
        self.delta_t = delta_t
        self._obs: dict = {}                           # frame_id → float64 xywh
        self._last_obs: np.ndarray = bbox.astype(np.float64)
        self.velocity: np.ndarray | None = None        # normalized (dx, dy)
        self._saved_state: np.ndarray | None = None    # statePost snapshot
        self._saved_cov: np.ndarray | None = None      # errorCovPost snapshot

    # ── KalmanTracker proxy ──────────────────────────────────────────────
    @property
    def tid(self) -> int:
        return self._kt.tid

    @property
    def hits(self) -> int:
        return self._kt.hits

    @property
    def time_since_update(self) -> int:
        return self._kt.time_since_update

    @time_since_update.setter
    def time_since_update(self, v: int) -> None:
        self._kt.time_since_update = v

    @property
    def pos(self) -> np.ndarray:
        return self._kt.pos

    def predict(self) -> np.ndarray:
        return self._kt.predict()

    def update(self, detection: np.ndarray) -> np.ndarray:
        return self._kt.update(detection)

    # ── OC-SORT methods ──────────────────────────────────────────────────
    def obs_center(self) -> np.ndarray:
        """Center (cx, cy) of the last real observation."""
        return self._last_obs[:2] + self._last_obs[2:] * 0.5

    def last_obs_int(self) -> np.ndarray:
        """Last real observation rounded to int64 xywh."""
        return np.round(self._last_obs).astype(np.int64)

    def record_obs(self, bbox: np.ndarray, frame_id: int) -> None:
        """Record a new detection, save Kalman posterior, update velocity."""
        self._last_obs = bbox.astype(np.float64)
        self._obs[frame_id] = self._last_obs.copy()
        # Prune to stay within memory bound
        max_keep = self.delta_t + 2
        if len(self._obs) > max_keep:
            for old_f in sorted(self._obs)[:-max_keep]:
                del self._obs[old_f]
        # Snapshot current posterior (after update() has been called)
        self._saved_state = self._kt.filter.statePost.copy()
        self._saved_cov = self._kt.filter.errorCovPost.copy()
        self._compute_velocity()

    def apply_oru(self, new_bbox: np.ndarray, frame_id: int) -> None:
        """ORU: restore saved state, replay with virtual observations."""
        if self._saved_state is None or not self._obs:
            return
        last_frame = max(self._obs)
        gap = frame_id - last_frame
        if gap <= 1:
            return
        # Restore Kalman posterior to last-seen frame
        self._kt.filter.statePost = self._saved_state.copy()
        self._kt.filter.errorCovPost = self._saved_cov.copy()
        last_obs_f = self._obs[last_frame]
        new_obs_f = new_bbox.astype(np.float64)
        # Replay K-1 predict+correct cycles with interpolated virtual obs
        for t in range(1, gap):
            alpha = t / gap
            virtual = np.round(
                (1.0 - alpha) * last_obs_f + alpha * new_obs_f
            ).astype(np.int64)
            self._kt.filter.predict()
            self._kt.filter.correct(au.tlwh2xyrh(virtual))
        # Final predict to reach the current frame
        self._kt.filter.predict()

    def _compute_velocity(self) -> None:
        if len(self._obs) < 2:
            self.velocity = None
            return
        curr_frame = max(self._obs)
        curr_c = self._center(self._obs[curr_frame])
        # Look for an observation at least delta_t frames back
        prev_obs = None
        for f in sorted(self._obs, reverse=True):
            if f < curr_frame - self.delta_t + 1:
                prev_obs = self._obs[f]
                break
        if prev_obs is None:
            earliest = min(self._obs)
            if earliest == curr_frame:
                self.velocity = None
                return
            prev_obs = self._obs[earliest]
        direction = curr_c - self._center(prev_obs)
        norm = float(np.linalg.norm(direction))
        self.velocity = (
            (direction / norm).astype(np.float64) if norm > 1e-6
            else np.zeros(2, dtype=np.float64)
        )

    @staticmethod
    def _center(bbox: np.ndarray) -> np.ndarray:
        return bbox[:2] + bbox[2:] * 0.5


class OCSORTracker:
    """OC-SORT multi-object tracker (motion-only, no appearance features).

    Parameters
    ----------
    min_hits : int
        Consecutive detections to confirm a new track.
    max_age : int
        Frames a lost track is kept before deletion.
    iou_threshold : float
        Minimum IoU to accept a match.
    inertia : float
        Weight of the OCM velocity-direction cost (0 = pure IoU).
    delta_t : int
        Observation window (frames) for velocity direction estimation.
    """

    def __init__(
        self,
        min_hits: int = 3,
        max_age: int = 30,
        iou_threshold: float = 0.3,
        inertia: float = 0.2,
        delta_t: int = 3,
        boxtype: argos.constants.OutlineStyle = argos.constants.OutlineStyle.bbox,
    ) -> None:
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.inertia = inertia
        self.delta_t = delta_t
        self.boxtype = boxtype
        self._tracks: dict = {}
        self._states: dict = {}
        self._next_id = 1
        self.frame_count = 0

    def reset(self) -> None:
        logging.debug('Resetting OCSORTracker.')
        self._tracks.clear()
        self._states.clear()
        self._next_id = 1
        self.frame_count = 0

    def update(self, bboxes: np.ndarray) -> dict:
        """Update tracker with new detections.

        Parameters
        ----------
        bboxes : np.ndarray
            Shape (N, 4) array in (x, y, w, h) format, dtype int64.

        Returns
        -------
        dict
            ``{track_id: bbox}`` for every confirmed track actively
            matched this frame.
        """
        self.frame_count += 1
        max_dist = 1.0 - self.iou_threshold

        # ── Predict ───────────────────────────────────────────────────────
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

        # ── Stage 1: IoU + OCM matching for active tracks ─────────────────
        if len(bboxes) > 0 and len(active_ids) > 0:
            cost = self._ocm_cost(bboxes[:, :KalmanTracker.NDIM], active_ids, predicted)
            matched1, unmatched_det_idx, unmatched_active = _hungarian(
                cost, active_ids, len(bboxes), max_dist
            )
        else:
            matched1 = {}
            unmatched_det_idx = set(range(len(bboxes)))
            unmatched_active = set(active_ids)

        # Update matched active tracks (with ORU if re-associating)
        for tid, det_i in matched1.items():
            track = self._tracks[tid]
            if track.time_since_update > 1:
                track.apply_oru(bboxes[det_i], self.frame_count)
            track.update(bboxes[det_i])
            track.record_obs(bboxes[det_i], self.frame_count)
            if (self._states[tid] == _TENTATIVE and
                    track.hits >= self.min_hits):
                self._states[tid] = _CONFIRMED

        # Unmatched active: tentative → delete, confirmed → lost
        for tid in list(unmatched_active):
            if self._states[tid] == _TENTATIVE:
                del self._tracks[tid]
                del self._states[tid]
            else:
                self._states[tid] = _LOST
                self._tracks[tid].time_since_update = 1

        # ── Stage 2 (OCR): last-observation IoU for lost tracks ───────────
        rem_list = sorted(unmatched_det_idx)
        if len(rem_list) > 0 and len(lost_ids) > 0:
            rem_bboxes = bboxes[rem_list, :KalmanTracker.NDIM]
            # Use actual last-observation positions (not drifted Kalman)
            last_obs_pred = {tid: self._tracks[tid].last_obs_int()
                             for tid in lost_ids}
            ocr_cost = _iou_cost(rem_bboxes, lost_ids, last_obs_pred, self.boxtype)
            matched2, unmatched_det2, unmatched_lost = _hungarian(
                ocr_cost, lost_ids, len(rem_bboxes), max_dist
            )
            for tid, rem_i in matched2.items():
                orig_i = rem_list[rem_i]
                track = self._tracks[tid]
                track.apply_oru(bboxes[orig_i], self.frame_count)
                track.update(bboxes[orig_i])
                track.record_obs(bboxes[orig_i], self.frame_count)
                self._states[tid] = _CONFIRMED
            for tid in unmatched_lost:
                if tid in self._tracks:
                    self._tracks[tid].time_since_update += 1
                    if self._tracks[tid].time_since_update > self.max_age:
                        del self._tracks[tid]
                        del self._states[tid]
            new_det_indices = [rem_list[i] for i in unmatched_det2]
        else:
            for tid in lost_ids:
                if tid in self._tracks:
                    self._tracks[tid].time_since_update += 1
                    if self._tracks[tid].time_since_update > self.max_age:
                        del self._tracks[tid]
                        del self._states[tid]
            new_det_indices = rem_list

        # ── New tentative tracks ──────────────────────────────────────────
        for det_i in new_det_indices:
            self._add_track(bboxes[det_i, :KalmanTracker.NDIM])

        # ── Return confirmed, actively-matched tracks ─────────────────────
        result: dict = {}
        for tid, track in self._tracks.items():
            if (self._states.get(tid) == _CONFIRMED
                    and track.time_since_update == 0
                    and (track.hits >= self.min_hits
                         or self.frame_count <= self.min_hits)):
                result[tid] = track.pos
        return result

    # ── Internal helpers ─────────────────────────────────────────────────

    def _ocm_cost(
        self,
        bboxes: np.ndarray,
        track_ids: list,
        pred_dict: dict,
    ) -> np.ndarray:
        """IoU cost + ``inertia`` × velocity-direction cost (OCM)."""
        labels = track_ids
        pred_arr = np.array(
            [pred_dict[tid] for tid in labels], dtype=np.int64
        )
        iou = pairwise_distance(
            bboxes.astype(np.int64), pred_arr,
            boxtype=self.boxtype,
            metric=argos.constants.DistanceMetric.iou,
        )
        if self.inertia <= 0:
            return iou
        return iou + self.inertia * self._vel_dir_cost(bboxes, labels)

    def _vel_dir_cost(
        self, bboxes: np.ndarray, track_ids: list
    ) -> np.ndarray:
        """OCM: arccos(direction·velocity) / π, shape (n_dets, n_tracks)."""
        n_dets = len(bboxes)
        n_tracks = len(track_ids)
        cost = np.zeros((n_dets, n_tracks), dtype=np.float64)
        det_centers = (
            bboxes[:, :2].astype(float) + bboxes[:, 2:4].astype(float) * 0.5
        )
        for j, tid in enumerate(track_ids):
            track = self._tracks[tid]
            if track.velocity is None:
                continue
            vel = track.velocity
            t_center = track.obs_center()
            dirs = det_centers - t_center          # (n_dets, 2)
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            valid = norms[:, 0] > 1e-6
            d_norm = np.where(
                valid[:, np.newaxis],
                dirs / np.where(valid[:, np.newaxis], norms, 1.0),
                0.0,
            )
            dots = d_norm @ vel
            cost[:, j] = np.arccos(np.clip(dots, -1.0, 1.0)) / np.pi
        return cost

    def _add_track(self, bbox: np.ndarray) -> None:
        track = _OCSortTrack(
            bbox, self._next_id, self.min_hits, self.max_age, self.delta_t
        )
        track.record_obs(bbox, self.frame_count)
        self._tracks[self._next_id] = track
        self._states[self._next_id] = _TENTATIVE
        self._next_id += 1


# ── Module-level helpers ─────────────────────────────────────────────────────

def _iou_cost(
    bboxes: np.ndarray,
    track_ids: list,
    pred_dict: dict,
    boxtype: argos.constants.OutlineStyle,
) -> np.ndarray:
    pred_arr = np.array([pred_dict[tid] for tid in track_ids], dtype=np.int64)
    return pairwise_distance(
        bboxes.astype(np.int64), pred_arr,
        boxtype=boxtype,
        metric=argos.constants.DistanceMetric.iou,
    )


def _hungarian(
    cost: np.ndarray,
    track_ids: list,
    n_dets: int,
    max_dist: float,
) -> tuple:
    """Run Hungarian assignment and return (matched, unmatched_dets, unmatched_tracks)."""
    if cost.size == 0:
        return {}, set(range(n_dets)), set(track_ids)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched: dict = {}
    good_rows: set = set()
    good_cols: set = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < max_dist:
            matched[track_ids[c]] = r
            good_rows.add(r)
            good_cols.add(track_ids[c])
    return (
        matched,
        set(range(n_dets)) - good_rows,
        set(track_ids) - good_cols,
    )
