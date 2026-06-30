# -*- coding: utf-8 -*-
"""Contour-derived shape descriptors used across the tracking pipeline.

All functions accept contour as np.ndarray of shape (N, 2) in (x, y) pixel
coordinates (float32 or int32).  They gracefully fall back to bbox-derived
values when the contour is None or has too few points.
"""
import numpy as np
import cv2

# Number of extra columns appended to the 4-element (x, y, w, h) bbox in the
# tracked-output dict.  Total track vector length = 4 + N_EXTRA = 13.
#   [4]   cx       – OBB centre x
#   [5]   cy       – OBB centre y
#   [6]   obb_w    – OBB width  (shorter side)
#   [7]   obb_h    – OBB height (longer side, i.e. body length)
#   [8]   angle    – OBB rotation angle in degrees (cv2 convention)
#   [9]   area     – filled contour area (px²)
#   [10]  major    – fitted ellipse major axis (px)
#   [11]  minor    – fitted ellipse minor axis (px)
#   [12]  solidity – area / convex-hull area  (0–1)
N_EXTRA = 9


def obb_from_contour(contour: np.ndarray | None,
                     bbox: np.ndarray) -> np.ndarray:
    """Return (cx, cy, obb_w, obb_h, angle) from contour, or from bbox."""
    x, y, w, h = bbox[:4]
    if contour is None or len(contour) < 5:
        return np.array([x + w / 2.0, y + h / 2.0, float(w), float(h), 0.0],
                        dtype=np.float64)
    cnt = contour.reshape(-1, 1, 2).astype(np.float32)
    (cx, cy), (bw, bh), angle = cv2.minAreaRect(cnt)
    # Convention: obb_h >= obb_w (body length is the longer axis)
    if bw > bh:
        bw, bh = bh, bw
        angle = (angle + 90.0) % 180.0
    return np.array([cx, cy, bw, bh, angle], dtype=np.float64)


def shape_metrics_from_contour(contour: np.ndarray | None,
                                bbox: np.ndarray) -> np.ndarray:
    """Return (area, major_axis, minor_axis, solidity) from contour, or bbox."""
    x, y, w, h = bbox[:4]
    if contour is None or len(contour) < 5:
        area = float(w) * float(h)
        major = float(max(w, h))
        minor = float(min(w, h))
        return np.array([area, major, minor, 1.0], dtype=np.float64)
    cnt = contour.reshape(-1, 1, 2).astype(np.float32)
    area = float(cv2.contourArea(cnt))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0.0 else 0.0
    try:
        _, (ma, mi), _ = cv2.fitEllipse(cnt)
        major, minor = float(max(ma, mi)), float(min(ma, mi))
    except cv2.error:
        _, (bw, bh), _ = cv2.minAreaRect(cnt)
        major, minor = float(max(bw, bh)), float(min(bw, bh))
    return np.array([area, major, minor, solidity], dtype=np.float64)


def extend_bbox(bbox: np.ndarray,
                contour: np.ndarray | None) -> np.ndarray:
    """Append OBB + shape metrics to a 4-element bbox array.

    Returns np.ndarray of shape (13,):
        [x, y, w, h,  cx, cy, obb_w, obb_h, angle,  area, major, minor, solidity]
    """
    obb = obb_from_contour(contour, bbox)
    metrics = shape_metrics_from_contour(contour, bbox)
    return np.concatenate([bbox[:4].astype(np.float64), obb, metrics])


def mask_iou(contour1: np.ndarray, contour2: np.ndarray) -> float:
    """IoU between two contours via rasterization.

    Both contours should be (N, 2) arrays in pixel coordinates.
    Returns 0.0 if either contour is degenerate.
    """
    c1 = contour1.reshape(-1, 2).astype(np.int32)
    c2 = contour2.reshape(-1, 2).astype(np.int32)
    if len(c1) < 3 or len(c2) < 3:
        return 0.0
    x_min = int(min(c1[:, 0].min(), c2[:, 0].min()))
    y_min = int(min(c1[:, 1].min(), c2[:, 1].min()))
    x_max = int(max(c1[:, 0].max(), c2[:, 0].max()))
    y_max = int(max(c1[:, 1].max(), c2[:, 1].max()))
    w, h = x_max - x_min + 1, y_max - y_min + 1
    if w <= 0 or h <= 0:
        return 0.0
    offset = np.array([[x_min, y_min]], dtype=np.int32)
    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m1, [c1 - offset], 1)
    cv2.fillPoly(m2, [c2 - offset], 1)
    inter = int(np.logical_and(m1, m2).sum())
    union = int(np.logical_or(m1, m2).sum())
    return float(inter) / float(union) if union > 0 else 0.0


def contour_iou_cost(bboxes: np.ndarray,
                     det_contours: list,
                     track_ids: list,
                     pred_bboxes: np.ndarray,
                     track_contours: dict,
                     boxtype,
                     metric) -> np.ndarray:
    """IoU cost matrix, using mask IoU when both sides have contours.

    Parameters
    ----------
    bboxes        : (N, 4) int64 detection bboxes (xywh)
    det_contours  : list of N contour arrays (or None)
    track_ids     : list of M track IDs
    pred_bboxes   : (M, 4) int64 predicted bboxes for each track_id
    track_contours: dict {tid: contour_or_None}
    boxtype, metric: passed to pairwise_distance for bbox fallback
    """
    from argos.utility import pairwise_distance
    # Base cost via bbox IoU
    cost = pairwise_distance(bboxes.astype(np.int64), pred_bboxes.astype(np.int64),
                             boxtype=boxtype, metric=metric)
    if not det_contours:
        return cost
    for i in range(len(bboxes)):
        d_cnt = det_contours[i] if i < len(det_contours) else None
        if d_cnt is None:
            continue
        for j, tid in enumerate(track_ids):
            t_cnt = track_contours.get(tid)
            if t_cnt is None:
                continue
            cost[i, j] = 1.0 - mask_iou(d_cnt, t_cnt)
    return cost
