# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-04-25 12:03 AM
"""Some utility functions for geometry"""
import enum
from typing import List
import sys
import cv2
import numpy as np
import logging

# Data type for rotated rectangle array
from PyQt5 import QtCore as qc, QtGui as qg

rotrect_dtype = np.dtype([('cx', float), ('cy', float),
                          ('w', float), ('h', float),
                          ('a', float)])
bbox_dtype = np.dtype([('x', int), ('y', int), ('w', int), ('h', int)])


# Enumeration for outline styles
class OutlineStyle(enum.Enum):
    bbox = enum.auto()
    minrect = enum.auto()
    contour = enum.auto()
    fill = enum.auto


class SegmentationMethod(enum.Enum):
    dbscan = enum.auto()
    threshold = enum.auto()
    watershed = enum.auto()


# Enumeration for distance metrics
class DistanceMetric(enum.Enum):
    iou = enum.auto()
    euclidean = enum.auto()


class TrackState(enum.Enum):
    tentative = enum.auto()
    confirmed = enum.auto()
    deleted = enum.auto()


def init():
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s '
                               'p=%(processName)s[%(process)d] '
                               't=%(threadName)s[%(thread)d] '
                               '%(filename)s#%(lineno)d:%(funcName)s: '
                               '%(message)s',
                        level=logging.DEBUG)

    qc.QCoreApplication.setOrganizationName('NIH')
    qc.QCoreApplication.setOrganizationDomain('nih.gov')
    qc.QCoreApplication.setApplicationName('Argos')

    settings = qc.QSettings()
    return settings

def rect_intersection(ra, rb):
    """Find if two axis-aligned rectangles intersect.

    ra, rb: rectangles specified as (x, y, w, h) where (x, y) is
    the coordinate of the lower left corner, w and h are width and height.

    This runs almost 50 times faster than Polygon intersection in shapely.
    and ~5 times faster than cv2.intersectConvexConvex.

    :return (x, y, dx, dy) specifying the overlap rectangle.
    If there is no overlap, all entries are 0.
    """
    xa, ya, wa, ha = ra
    xb, yb, wb, hb = rb
    x = max(xa, xb)
    y = max(ya, yb)
    dx = min(xa + wa, xb + wb) - x
    dy = min(ya + ha, yb + hb) - y
    if (dx > 0) and (dy > 0):
        return (x, y, dx, dy)
    return (0, 0, 0, 0)


def rect_iou(ra, rb):
    """
    Compute Intersection over Union of two axis-aligned rectangles.
    This is the ratio of the are of intersection to the area of the union
    of the two rectangles.

    ra, rb: two axis aligned rectangles specified as (x, y, w, h) where
    (x, y) is the position of the lower left corner, w and h are width
    and height.

    :return the Intersection over Union of two rectangles.
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


def run_fn(arg0: np.ndarray, args: List) -> np.ndarray:
    """Run functions and arguments on args.

    arg0 is the first argument to the first function. All the following
    functions take the output of the previous one as the first argument.

    args should be [(fn, positional_args, kwargs), ...]

    :returns the processed _image data as a `numpy.ndarray`
    """
    result = None
    for fn, pargs, kwargs in args:
        logging.debug('Running fn %r, nargs: %d, kwargs: %r', fn, len(args),
                      kwargs.keys())
        if result is not None:
            result = fn(result, *pargs, **kwargs)
        else:  # first function needs arg0
            result = fn(arg0, *pargs, **kwargs)
        logging.debug(f'Result: len {len(result)} of type {type(result)}')
    return result


def rectpoly_points(p0: tuple, p1: tuple) -> tuple:
    """Generate four clockwise corner positions starting from
    top left of the rectangle with any pair of diagonal points `p0` and `p1`"""
    x = p0[0], p1[0]
    y = p0[1], p1[1]
    xleft = min(x)
    w = max(x) - xleft
    ytop = min(y)
    h = max(y) - ytop
    return rect2points(xleft, ytop, w, h)


def rect2points(xleft, ytop, w, h):
    """Convert topleft, width, height format rectangle into four clockwise
    vertices"""
    return ((xleft, ytop), (xleft, ytop + h),
            (xleft + w, ytop + h), (xleft + w, ytop))


def points2rect(p0: qc.QPointF, p1: qc.QPointF) -> qc.QRectF:
    """:returns a rectangle with diagonal corners `p0` and `p1`
    after scaling by `scale`. This will work with both top-left - bottom-right
    and bottom-left - top-right diagonals.
    """
    x = p0.x(), p1.x()
    y = p0.y(), p1.y()
    xleft = min(x)
    w = max(x) - xleft
    ytop = min(y)
    h = max(y) - ytop
    return qc.QRectF(xleft, ytop, w, h)


def poly2xyrh(vtx):
    """Convert clockwise vertices into centre, aspect ratio, height format"""
    x0, y0 = vtx[0]
    x1, y1 = vtx[2]
    w = x1 - x0
    h = y1 - y0
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0, w / float(h), float(h)


def tlwh2xyrh(x, y, w, h):
    """Convert top-left, width, height into center, aspect ratio, height"""
    return x +  w / 2.0, y + h / 2.0, w / float(h), float(h)


def xyrh2tlwh(x, y, r, h):
    """Convert centre, aspect ratio, height into top-left, width, height
    format"""
    w = r * h
    return x - w / 2.0, y - h / 2.0, w, h


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


