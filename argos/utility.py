# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-04-25 12:03 AM
"""Some utility functions for geometry"""
from typing import Tuple, List, Dict, Set
import sys
import cv2
import numpy as np
import logging
from matplotlib import cm
from math import floor

# Data type for rotated rectangle array
from PyQt5 import QtCore as qc, QtGui as qg
from scipy import optimize
from sklearn.utils.murmurhash import murmurhash3_32

from argos.constants import OutlineStyle, DistanceMetric


def init():
    qc.QCoreApplication.setOrganizationName('NIH')
    qc.QCoreApplication.setOrganizationDomain('nih.gov')
    qc.QCoreApplication.setApplicationName('Argos')

    settings = qc.QSettings()
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s '
                               'p=%(processName)s[%(process)d] '
                               't=%(threadName)s[%(thread)d] '
                               '%(filename)s#%(lineno)d:%(funcName)s: '
                               '%(message)s',
                        level=logging.DEBUG)
    return settings


def rectpoly_points(p0: tuple, p1: tuple) -> tuple:
    """Generate four clockwise corner positions starting from
    top left of the rectangle with any pair of diagonal points `p0` and `p1`"""
    x = p0[0], p1[0]
    y = p0[1], p1[1]
    xleft = min(x)
    w = max(x) - xleft
    ytop = min(y)
    h = max(y) - ytop
    return rect2points(np.array((xleft, ytop, w, h)))


def points2rect(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """:returns a rectangle with diagonal corners `p0` and `p1`
    after scaling by `scale`. This will work with both top-left - bottom-right
    and bottom-left - top-right diagonals.
    """
    x = p0[0], p1[0]
    y = p0[1], p1[1]
    xleft = min(x)
    w = max(x) - xleft
    ytop = min(y)
    h = max(y) - ytop
    return np.array((xleft, ytop, w, h))


def poly2xyrh(vtx):
    """Convert clockwise vertices into centre, aspect ratio, height format"""
    x0, y0 = vtx[0]
    x1, y1 = vtx[2]
    w = x1 - x0
    h = y1 - y0
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0, w / float(h), float(h)


def to_qpolygon(points, scale=1.0):
    return qg.QPolygonF(
        [qc.QPointF(p0 * scale, p1 * scale) for p0, p1 in points])


def cond_bbox_overlap(ra, rb, min_iou):
    """Check if IoU of axis-aligned bounding boxes is at least `min_iou`
    ra, rb: rectangles specified as (x, y, w, h)
    """
    return rect_iou(ra, rb) >= min_iou


def cond_minrect_overlap(ra, rb, min_iou):
    """Check if IoU of minimum area (rotated) bounding rectangles is at least `min_iou`.
    ra, rb: box points with coordinates of the four corners.
    """
    area_i, _ = cv2.intersectConvexConvex(ra, rb)
    area_u = cv2.contourArea(ra) + cv2.contourArea(rb) - area_i
    return area_i >= min_iou * area_u  # Avoids divide by zero


def cond_proximity(points_a, points_b, min_dist):
    """Check if the proximity of two arrays of points is more than `min_dist`.

    To take the shape of the object into account, I use the following measure
    of distance:
    scale the distance between centres of mass by the geometric mean of the
    square roots of the second moments.

    (x1 - x2) / sqrt(sigma_1_x * sigma_2_x)
    (y1 - y2) / sqrt(sigma_1_y * sigma_2_y)
    """
    sigma = np.std(points_a, axis=0) * np.std(points_b, axis=0)
    dx2 = (np.mean(points_a[:, 0]) - np.mean(points_b[:, 0])) ** 2 / sigma[0]
    dy2 = (np.mean(points_a[:, 1]) - np.mean(points_b[:, 1])) ** 2 / sigma[1]
    return dx2 + dy2 < min_dist ** 2


def cv2qimage(frame: np.ndarray, copy: bool = False) -> qg.QImage:
    """Convert BGR/gray/bw frame into QImage"""
    if (len(frame.shape) == 3) and (frame.shape[2] == 3):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        qimg = qg.QImage(img.tobytes(), w, h, w * c, qg.QImage.Format_RGB888)
    elif len(frame.shape) == 2:  # grayscale
        h, w = frame.shape
        qimg = qg.QImage(frame.tobytes(), w, h, w * 1,
                         qg.QImage.Format_Grayscale8)
    if copy:
        return qimg.copy()
    return qimg


# Try to import cython version of these functions, otherwise define them in
# pure python
try:
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()},
                      reload_support=True)
    from cutility import (rect2points, tlwh2xyrh, xyrh2tlwh,
                          rect_intersection,
                          rect_iou,
                          pairwise_distance)
except:
    def rect2points(rect: np.ndarray) -> np.ndarray:
        """Convert topleft, width, height format rectangle into four anti-clockwise
        vertices"""
        return np.vstack([rect[:2],
                          (rect[0], rect[1] + rect[3]),
                          rect[:2] + rect[2:],
                          (rect[0] + rect[2], rect[1])])


    def tlwh2xyrh(rect):
        """Convert top-left, width, height into center, aspect ratio, height"""
        return np.array((rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0,
                         rect[2] / float(rect[3]), rect[3]))


    def xyrh2tlwh(rect: np.ndarray) -> np.ndarray:
        """Convert centre, aspect ratio, height into top-left, width, height
        format"""
        w = rect[2] * rect[3]
        return np.asanyarray((rect[0] - w / 2.0, rect[1] - rect[3] / 2.0,
                              w, rect[3]),
                             dtype=int)


    def rect_intersection(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
        """Find if two axis-aligned rectangles intersect.

        This runs almost 50 times faster than Polygon intersection in shapely.
        and ~5 times faster than cv2.intersectConvexConvex.

        Parameters
        ----------
        ra: np.ndarray
        rb: np.ndarray
            Rectangles specified as (x, y, w, h) where (x, y) is the coordinate
            of the lower left corner, w and h are width and height.

        Returns
        -------
        np.ndarray
            (x, y, dx, dy) specifying the overlap rectangle. If there is no
            overlap, all entries are 0.
        """
        ret = np.zeros((4,), dtype=int)
        xa, ya, wa, ha = ra
        xb, yb, wb, hb = rb
        x = max(xa, xb)
        y = max(ya, yb)
        dx = min(xa + wa, xb + wb) - x
        dy = min(ya + ha, yb + hb) - y
        if (dx > 0) and (dy > 0):
            ret[:] = (x, y, dx, dy)
        return ret


    def rect_iou(ra: np.ndarray, rb: np.ndarray) -> float:
        """Compute Intersection over Union of two axis-aligned rectangles.

        This is the ratio of the are of intersection to the area of the union
        of the two rectangles.

        Parameters
        ----------
        ra: np.ndarray
        rb: np.ndarray
            Axis aligned rectangles specified as (x, y, w, h) where (x, y) is
            the position of the lower left corner, w and h are width and height.

        Returns
        -------
        float
            The Intersection over Union of two rectangles.
        """
        x, y, dx, dy = rect_intersection(ra, rb)
        area_i = dx * dy
        area_u = ra[2] * ra[3] + rb[2] * rb[3] - area_i
        if area_u <= 0 or area_i < 0:
            raise ValueError('Area not positive')
        ret = 1.0 * area_i / area_u
        if np.isinf(ret) or np.isnan(ret) or ret < 0:
            raise ValueError('Invalid intersection')
        return ret


    def pairwise_distance(new_bboxes: np.ndarray, bboxes: np.ndarray,
                          boxtype: OutlineStyle,
                          metric: DistanceMetric) -> np.ndarray:
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
        if metric == DistanceMetric.euclidean:
            centers = bboxes[:, :2] + bboxes[:, 2:] * 0.5
            new_centers = new_bboxes[:, :2] + new_bboxes[:, 2:] * 0.5
            for ii in range(len(new_bboxes)):
                for jj in range(len(bboxes)):
                    dist[ii, jj] = np.sum((new_centers[ii] - centers[jj]) ** 2)
        elif metric == DistanceMetric.iou:
            if boxtype == OutlineStyle.bbox:  # This can be handled efficiently
                for ii in range(len(new_bboxes)):
                    for jj in range(len(bboxes)):
                        dist[ii, jj] = 1.0 - rect_iou(bboxes[jj],
                                                      new_bboxes[ii])
            else:
                raise NotImplementedError(
                    'Only handling axis-aligned bounding boxes')
        else:
            raise NotImplementedError(f'Unknown metric {metric}')
        return dist


def match_bboxes(id_bboxes: dict, new_bboxes: np.ndarray,
                 boxtype: OutlineStyle,
                 metric: DistanceMetric = DistanceMetric.euclidean,
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
    logging.debug('Current bboxes:\n%s', '\n'.join([str(i) for i in id_bboxes.items()]))
    logging.debug('New bboxes:\n%r', new_bboxes)
    logging.debug('Box type: %r', boxtype)
    logging.debug('Max dist: %r', max_dist)
    if len(id_bboxes) == 0:
        return ({}, set(range(len(new_bboxes))), {})
    labels = list(id_bboxes.keys())
    bboxes = np.array(list(id_bboxes.values()), dtype=float)
    dist_matrix = pairwise_distance(new_bboxes, bboxes, boxtype=boxtype,
                                    metric=metric)
    row_ind, col_ind = optimize.linear_sum_assignment(dist_matrix)
    matched = {}
    good_rows = set()
    good_cols = set()
    if metric == DistanceMetric.euclidean:
        max_dist *= max_dist
    for row, col in zip(row_ind, col_ind):
        if dist_matrix[row, col] < max_dist:
            good_rows.add(row)
            good_cols.add(labels[col])
            matched[labels[col]] = row
    new_unmatched = set(range(len(new_bboxes))) - good_rows
    old_unmatched = set(id_bboxes.keys()) - good_cols
    return matched, new_unmatched, old_unmatched


def reconnect(signal, newhandler=None, oldhandler=None):
    """Disconnect signal from oldhandler and connect to newhandler"""
    while True:
        try:
            if oldhandler is not None:
                signal.disconnect(oldhandler)
            else:
                signal.disconnect()
        except TypeError:
            break
    if newhandler is not None:
        signal.connect(newhandler)


def make_color(num: int) -> Tuple[int]:
    """Create a random color based on number.

    The provided number is passed through the murmur hash function in order
    to generate bytes which are somewhat apart from each other. The three least
    significant byte values are taken as r, g, and b.

    Parameters
    ----------
    num: int
        number to use as hash key

    Returns
    -------
    bytes[3]
        (r, g, b) values

    """
    val = murmurhash3_32(num, positive=True).to_bytes(8, 'little')
    color = qg.QColor(val[0], val[1], val[2])
    return val[:3]


def get_cmap_color(num, maxnum, cmap):
    """Get rgb based on specified colormap `cmap` for index `num` where the
    total range of values is (0, maxnum]"""
    rgba = cm.get_cmap(cmap)(float(num) / maxnum)
    int_rgb = [max(0, min(255, floor(v * 256))) for v in rgba[:3]]
    return int_rgb
