# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-04-25 12:03 AM
"""
=========================
General utility functions
=========================

"""
import os
from typing import Tuple, List, Dict, Set
import sys
import cv2
import numpy as np
import logging
from matplotlib import cm
from math import floor

# Data type for rotated rectangle array
from PyQt5 import QtCore as qc, QtGui as qg, QtWidgets as qw
from scipy import optimize
from sklearn.utils.murmurhash import murmurhash3_32

from argos.constants import OutlineStyle, DistanceMetric



class WaitingSpinner(qw.QWidget):
    mColor = qg.QColor(qc.Qt.gray)
    mRoundness = 100.0
    mMinimumTrailOpacity = 31.4159265358979323846
    mTrailFadePercentage = 50.0
    mRevolutionsPerSecond = 1.57079632679489661923
    mNumberOfLines = 20
    mLineLength = 10
    mLineWidth = 2
    mInnerRadius = 20
    mCurrentCounter = 0
    mIsSpinning = False

    def __init__(self, centerOnParent=True, disableParentWhenSpinning=True, *args, **kwargs):
        super(WaitingSpinner, self).__init__(*args, **kwargs)
        self.mCenterOnParent = centerOnParent
        self.mDisableParentWhenSpinning = disableParentWhenSpinning
        self.initialize()

    def initialize(self):
        self.timer = qc.QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.updateSize()
        self.updateTimer()
        self.hide()

    @qc.pyqtSlot()
    def rotate(self):
        self.mCurrentCounter += 1
        if self.mCurrentCounter > self.numberOfLines():
            self.mCurrentCounter = 0
        self.update()

    def updateSize(self):
        size = (self.mInnerRadius + self.mLineLength) * 2
        self.setFixedSize(size, size)

    def updateTimer(self):
        self.timer.setInterval(1000 / (self.mNumberOfLines *
                                       self.mRevolutionsPerSecond))

    def updatePosition(self):
        if self.parentWidget() and self.mCenterOnParent:
            self.move(self.parentWidget().width() / 2 - self.width() / 2,
                      self.parentWidget().height() / 2 - self.height() / 2)

    def lineCountDistanceFromPrimary(self, current, primary, totalNrOfLines):
        distance = primary - current
        if distance < 0:
            distance += totalNrOfLines
        return distance

    def currentLineColor(self, countDistance, totalNrOfLines, trailFadePerc,
                         minOpacity, color):
        if countDistance == 0:
            return color

        minAlphaF = minOpacity / 100.0

        distanceThreshold = np.ceil((totalNrOfLines - 1) * trailFadePerc / 100.0)
        if countDistance > distanceThreshold:
            color.setAlphaF(minAlphaF)

        else:
            alphaDiff = self.mColor.alphaF() - minAlphaF
            gradient = alphaDiff / distanceThreshold + 1.0
            resultAlpha = color.alphaF() - gradient * countDistance
            resultAlpha = min(1.0, max(0.0, resultAlpha))
            color.setAlphaF(resultAlpha)
        return color

    def paintEvent(self, event):
        self.updatePosition()
        painter = qg.QPainter(self)
        painter.fillRect(self.rect(), qc.Qt.transparent)
        painter.setRenderHint(qg.QPainter.Antialiasing, True)
        if self.mCurrentCounter > self.mNumberOfLines:
            self.mCurrentCounter = 0
        painter.setPen(qc.Qt.NoPen)

        for i in range(self.mNumberOfLines):
            painter.save()
            painter.translate(self.mInnerRadius + self.mLineLength,
                              self.mInnerRadius + self.mLineLength)
            rotateAngle = 360.0 * i / self.mNumberOfLines
            painter.rotate(rotateAngle)
            painter.translate(self.mInnerRadius, 0)
            distance = self.lineCountDistanceFromPrimary(i,
                                                         self.mCurrentCounter,
                                                         self.mNumberOfLines)
            color = self.currentLineColor(distance, self.mNumberOfLines,
                                          self.mTrailFadePercentage,
                                          self.mMinimumTrailOpacity,
                                          self.mColor)
            painter.setBrush(color)
            painter.drawRoundedRect(qg.QRect(0, -self.mLineWidth // 2,
                                             self.mLineLength, self.mLineLength),
                                    self.mRoundness, qc.Qt.RelativeSize)
            painter.restore()

    def start(self):
        self.updatePosition()
        self.mIsSpinning = True
        self.show()

        if self.parentWidget() and self.mDisableParentWhenSpinning:
            self.parentWidget().setEnabled(False)

        if not self.timer.isActive():
            self.timer.start()
            self.mCurrentCounter = 0

    def stop(self):
        self.mIsSpinning = False
        self.hide()

        if self.parentWidget() and self.mDisableParentWhenSpinning:
            self.parentWidget().setEnabled(True)

        if self.timer.isActive():
            self.timer.stop()
            self.mCurrentCounter = 0


def init():
    """Initialize logging and Qt settings."""
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


def to_qpolygon(points, scale=1.0):
    """Convert a sequence of (x, y) points into a `qg.QPolygonF`.

    """
    return qg.QPolygonF(
        [qc.QPointF(p0 * scale, p1 * scale) for p0, p1 in points])


def cond_bbox_overlap(ra, rb, min_iou):
    """Check if IoU of axis-aligned bounding boxes overlap.

    Parameters
    ----------
    ra, rb: array like
        Rectangles specified as (x, y, w, h)
    min_iou: flat
        Minimum value of IoU to consider overlap.

    Returns
    -------
    bool
        True if `ra` and `rb` have IoU >= `min_iou`. False otherwise.
    """
    return rect_iou(ra, rb) >= min_iou


def cond_minrect_overlap(ra, rb, min_iou):
    """Check if IoU of minimum area (rotated) bounding rectangles is at least
    `min_iou`.

    Parameters
    ----------
    ra: array like
        First rectangle defined by the coordinates of four corners.
    rb: array like
        Second rectangle defined by the coordinates of four corners.
    min_iou: float
        Minimum overlap defined by intersection over union of bounding boxes.

    Returns
    -------
    bool
        True if area of overlap is greater or equal to `min_iou`.
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

    Parameters
    ----------
    points_a: array like
        Sequence of points
    points_b: array like
        Sequence of points
    min_dist: float
        Minimum distance.

    Returns
    -------
    bool
        `True` if the centres of mass (mean position) of `points_a` and
        `points_b` are closer than `min_dist`, `False` otherwise.
    """
    sigma = np.std(points_a, axis=0) * np.std(points_b, axis=0)
    dx2 = (np.mean(points_a[:, 0]) - np.mean(points_b[:, 0])) ** 2 / sigma[0]
    dy2 = (np.mean(points_a[:, 1]) - np.mean(points_b[:, 1])) ** 2 / sigma[1]
    return dx2 + dy2 < min_dist ** 2


def cv2qimage(frame: np.ndarray, copy: bool = False) -> qg.QImage:
    """Convert BGR/gray/bw frame from array into QImage".
    
    OpenCV reads images into 2D or 3D matrix. This function converts it into
    Qt QImage.

    Parameters
    ----------
    frame: numpy.ndarray
        Input image data as a 2D (black and white, gray() or 3D (color, OpenCV
        reads images in BGR instead of RGB format) array.
    copy: bool, default False
        If True Make a copy of the image data.

    Returns
    -------
    qg.QImage
        Converted image.
    """
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
    # import pyximport

    # pyximport.install(setup_args={"include_dirs": np.get_include()},
    #                  reload_support=True)
    from argos.cutility import (rect2points, tlwh2xyrh, xyrh2tlwh,
                                rect_intersection,
                                rect_iou,
                                rect_ios,
                                pairwise_distance)

    logging.info('Loaded C-utilities with pyximport')
    print('Loaded C-utilities with pyximport')
except ImportError as err:
    print('Could not load C-utilities with pyximport. Using pure Python.')
    print(err)
    logging.info(
        'Could not load C-utilities with pyximport. Using pure Python.')
    logging.info(f'{err}')


    def points2rect(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Convert diagonally opposite vertices into (x, y, w, h) format
        rectangle.

        Returns
        -------
        np.ndarray:
            Rectangle with diagonal corners `p0` and `p1` after scaling
            by `scale`. This will work with both top-left - bottom-right
            and bottom-left - top-right diagonals.

        """
        x = p0[0], p1[0]
        y = p0[1], p1[1]
        xleft = min(x)
        w = max(x) - xleft
        ytop = min(y)
        h = max(y) - ytop
        return np.array((xleft, ytop, w, h))


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


    def rect_ios(ra: np.ndarray, rb: np.ndarray) -> float:
        """Compute intersection over area of smaller of two axis-aligned
        rectangles.

        This is the ratio of the area of intersection to the area of the smaller
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
            The Intersection over area of the smaller of two rectangles.
        """
        x, y, dx, dy = rect_intersection(ra, rb)
        area_i = dx * dy
        area_a = ra[2] * ra[3]
        area_b = rb[2] * rb[3]
        if area_i < 0 or area_a <= 0 or area_b <= 0:
            raise ValueError('Area not positive')
        ret = area_i / min(area_a, area_b)
        if np.isinf(ret) or np.isnan(ret) or ret < 0:
            raise ValueError('Invalid intersection')
        return ret


    def pairwise_distance(new_bboxes: np.ndarray, bboxes: np.ndarray,
                          boxtype: OutlineStyle,
                          metric: DistanceMetric) -> np.ndarray:
        """Computes the distance between all pairs of rectangles.

        Parameters
        ----------
        new_bboxes: np.ndarray
           Array of bounding boxes, each row as (x, y, w, h)
        bboxes: np.ndarray
           Array of bounding boxes, each row as (x, y, w, h)
        boxtype: {OutlineStyle.bbox, OulineStyle.minrect}
           OutlineStyle.bbox for axis aligned rectangle bounding box or
           OulineStyle.minrect for minimum area rotated rectangle
        metric: {DistanceMetric.euclidean, DistanceMetric.iou}
           When `DistanceMetric.euclidean`, the squared Euclidean distance is
           used (calculating square root is expensive and unnecessary. If
           `DistanceMetric.iou`, use the area of intersection divided by the
           area of union.

        Returns
        --------
        np.ndarray
            Row ``ii``, column ``jj`` contains the computed distance between
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
        elif metric == DistanceMetric.ios and boxtype == OutlineStyle.bbox:
            for ii in range(len(new_bboxes)):
                for jj in range(len(bboxes)):
                    dist[ii, jj] = 1.0 - rect_ios(bboxes[jj],
                                                  new_bboxes[ii])
        else:
            raise NotImplementedError(f'Unknown metric {metric}')
        return dist


def match_bboxes(id_bboxes: dict, new_bboxes: np.ndarray,
                 boxtype: OutlineStyle,
                 metric: DistanceMetric = DistanceMetric.euclidean,
                 max_dist: float = 10000
                 ) -> Tuple[Dict[int, int], Set[int], Set[int]]:
    """Match the rectangular bounding boxes in `new_bboxes` to the closest
    object in the `id_bboxes` dictionary.

    Parameters
    ----------
    id_bboxes: dict[int, np.ndarray]
        Mapping ids to bounding boxes
    new_bboxes: np.ndarray
        Array of new bounding boxes to be matched to those in ``id_bboxes``.
    boxtype: {OutlineStyle.bbox, OutlineStyle.minrect}
        Type of bounding box to match.
    max_dist: int, default 10000
        Anything that is more than this distance from all of the bboxes in
        ``id_bboxes`` are put in the unmatched list
    metric: {DistanceMetric.euclidean, DistanceMetric.iou}
        `DistanceMetric.euclidean` for Euclidean distance between centers of the
        boxes. `DistanceMetric.iou` for area of inetersection over union of the
        boxes,

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
    # logging.debug('Current bboxes:\n%s', '\n'.join([str(i) for i in id_bboxes.items()]))
    # logging.debug('New bboxes:\n%r', new_bboxes)
    # logging.debug('Box type: %r', boxtype)
    # logging.debug('Max dist: %r', max_dist)
    if len(id_bboxes) == 0:
        return ({}, set(range(len(new_bboxes))), {})
    labels = list(id_bboxes.keys())
    bboxes = np.array(np.rint(list(id_bboxes.values())), dtype=np.int_)
    dist_matrix = pairwise_distance(new_bboxes, bboxes, boxtype=boxtype,
                                    metric=metric)
    row_ind, col_ind = optimize.linear_sum_assignment(dist_matrix)
    if metric == DistanceMetric.euclidean:
        max_dist *= max_dist
    result = [(row, col, (labels[col], row))
              for row, col in zip(row_ind, col_ind)
              if dist_matrix[row, col] < max_dist]
    if len(result) > 0:
        good_rows, good_cols, matched = zip(*result)
        good_rows = set(good_rows)
        good_cols = set(good_cols)
        matched = dict(matched)
        new_unmatched = set(range(len(new_bboxes))) - good_rows
        old_unmatched = set(id_bboxes.keys()) - good_cols
    else:
        matched = {}
        new_unmatched = set(range(len(new_bboxes)))
        old_unmatched = set(id_bboxes.keys())
    return matched, new_unmatched, old_unmatched


def reconnect(signal, newhandler=None, oldhandler=None):
    """Disconnect PyQt signal from oldhandler and connect to newhandler"""
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
    # color = qg.QColor(val[0], val[1], val[2])
    return val[:3]


def get_cmap_color(num, maxnum, cmap):
    """Get rgb based on specified colormap `cmap` for index `num` where the
    total range of values is (0, maxnum].

    Parameters
    ----------
    num: real number
        Position into colormap.
    maxnum: real number
        Normalize `num` by this value.
    cmap: str
        Name of colormap

    Returns
    -------
    tuple: (r, g, b)
        The red, green and blue value for the color at position `num`/`maxnum`
        in the (0, 1) range of the colormap.
    """
    rgba = cm.get_cmap(cmap)(float(num) / maxnum)
    int_rgb = (max(0, min(255, floor(v * 256))) for v in rgba[:3])
    return int_rgb


def extract_frames(vidfile, nframes, scale=1.0, outdir='.', random=False):
    """Extract `nframes` frames from `vidfile` into `outdir`"""
    cap = cv2.VideoCapture(vidfile)
    fname = os.path.basename(vidfile)
    prefix = fname.rpartition('.')[0]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.arange(frame_count, dtype=int)
    if frame_count < nframes:
        if random:
            np.random.shuffle(idx)
            idx = idx[:nframes]
        else:
            step = frame_count // nframes
            idx = idx[::step]
    idx = sorted(idx)
    ii = 0
    jj = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        if idx[jj] == ii:
            size = (int(frame.shape[1] * scale),
                int(frame.shape[0] * scale))
            if scale < 1:
                frame = cv2.resize(frame, size, cv2.INTER_AREA)
            elif scale > 1:
                frame = cv2.resize(frame, size, cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(outdir,
                                     f'{prefix}_{idx[jj]:06d}.png'), frame)
            jj += 1
        ii += 1
    cap.release()
