# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-11-27 8:34 PM
import logging
from typing import List, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore as qc
from sklearn import cluster

from argos import utility as ut


def segment_by_dbscan(binary_img: np.ndarray, eps: float=5,
                      min_samples: int=10) -> List[np.ndarray]:
    """Use DBSCAN clustering to segment binary image.

    Parameters
    ----------
    binary_img: np.ndarray
        binary image, a 2D array containing 0s and 1s (obtaind by thresholding
        original image converted to grayscale).
    eps: float
        the epsilon parameter of DBSCAN.
    min_samples: int
        minimum number of pixels each cluster (object) must contain in order to
        be considered a valid object.

    Returns
    -------
    list
        List of coordinate arrays where the n-th entry is the array of
        positions of the pixels belonging to the n-th segmented object.
    """
    indices = np.nonzero(binary_img)
    xy = np.vstack((indices[1], indices[0])).T
    core, labels = cluster.dbscan(xy, eps=eps, min_samples=min_samples,
                                  metric='euclidean', algorithm='auto')
    unique_labels = set(labels)
    unique_labels.discard(-1)  # -1 is the noise label
    return [xy[labels == label] for label in sorted(unique_labels)]


def segment_by_contours(binary_img: np.ndarray) -> List[np.ndarray]:
    """Segment binary image by finding contours of contiguous
    nonzero pixels and then filling those contours with an integer
    color value.

    Although, this is also part of the watershed algorithm, for small objects
    that are clearly separable from the background and each other, this
    works equally well.

    Parameters
    ----------
    binary_img: numpy.ndarray
        binary input image (obtained by thresholding grayscale image).

    Returns
    -------
    list
        List of coordinate arrays where the n-th entry is the array of
        positions of the pixels belonging to the n-th segmented object.
    """
    contours, hierarchy = cv2.findContours(binary_img,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # logging.debug(f'Segmented {len(contours)} objects.')
    segmented = np.zeros(binary_img.shape, dtype=np.int32)
    for ii, contour in enumerate(contours):
        cv2.drawContours(segmented, [contour], -1, thickness=cv2.FILLED,
                         color=ii + 1)
    unique_labels = set(segmented.flat)
    unique_labels.discard(0)
    ret = [np.argwhere(segmented == label) for label in sorted(unique_labels)]
    # Fast swapping of y and x - see answer by blax here:
    # https://stackoverflow.com/questions/4857927/swapping-columns-in-a-numpy-array
    for points in ret:
        points[:, 0], points[:, 1] = points[:, 1], points[:, 0].copy()
    return ret


def segment_by_watershed(binary_img: np.ndarray, img: np.ndarray,
                         dist_thresh: float=3.0) -> Tuple[np.ndarray,
                                                          List[np.ndarray]]:
    """Segment image using watershed algorithm.

    Parameters
    ----------
    binary_img:np.ndarray
        Binary image derived from ``img`` with nonzero pixel blobs for objects.
        This is usually produced after converting the ``img`` to grayscale and
        then thresholding.
    img: np.ndarray
        Original image to be segmented.
    dist_thresh: float, optional
        Threshold for distance of pixels from boundary to consider them core
        points. If it is < 1.0, it is interpreted as fraction of the maximum
        of the distances.

    Returns
    -------
    points_list: list[np.ndarray]
        List of arrays containing positions of the pixels in each object.

    Notes
    -----
    This code is derivative of this OpenCV tutorial:

    https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html

    and falls under the same licensing.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
    # Distance transform calculates the distance of each pixel from
    # background (black in this case) pixels. So we have an image
    # where pixel intensity means the distance of that pixel from the
    # background
    dist_xform = cv2.distanceTransform(opening, cv2.DIST_L2,
                                       cv2.DIST_MASK_PRECISE)
    # Thresholding the distance image to find pixels which are more
    # than a certain distance away from background - this should give
    # us the pixels central to foreground objects
    if dist_thresh < 1.0:
        # threshold relative to maximum of computed distance
        dist_thresh *= dist_xform.max()
    ret, sure_fg = cv2.threshold(dist_xform, dist_thresh, 255, 0)
    sure_fg = np.uint8(sure_fg)
    # border between background and foreground
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg, connectivity=4)
    # logging.debug(f'Found {ret} connected components')
    # 0 is for background - assign a large value to keep them off
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    unique_labels = set(markers.flat)
    unique_labels.discard(-1)
    unique_labels.discard(0)
    ret = [np.argwhere(markers == label) for label in sorted(unique_labels)]
    # markers[markers == -1] = 0
    # markers = np.uint8(markers)
    # Fast swapping of y and x - see answer by blax here:
    # https://stackoverflow.com/questions/4857927/swapping-columns-in-a-numpy-array
    for points in ret:
        points[:, 0], points[:, 1] = points[:, 1], points[:, 0].copy()
    return ret


def extract_valid(points_list, pmin, pmax, wmin, wmax, hmin, hmax, roi=None):
    """
    Filter valid objects based on size limits.

    The length of the smaller side of the minimum bounding rotated-rectangle is
    considered width and the larger as height.

    Parameters
    ----------
    points_list: list[np.ndarray]
        List of coordinate arrays for pixels in each segmented object pixels.
    pmin: int
        Minimum number of pixels.
    pmax: int
        Maximum number of pixels.
    wmin: int
        Minimum width of minimum bounding rotated rectangle.
    wmax: int
        Maximum width of minimum bounding rotated rectangle.
    hmin: int
        Minimum height/length of minimum bounding rotated rectangle.
    hmax: int
        Maximum height/length of minimum bounding rotated rectangle.

    Returns
    -------
    list
        Coordinate arrays of objects that are between ``pmin`` and ``pmax``
        pixels, ``wmin`` and ``wmax`` width, and ``hmin`` and ``hmax`` height
        where The length of the smaller side of the minimum bounding
        rotated-rectangle is considered width and the larger as height.
    """
    # logging.debug(f'Parameters: pmin {pmin}, pmax {pmax}, wmin {wmin}, '
    #               f'wmax {wmax}, hmin {hmin}, hmax {hmax}')
    minrects = [cv2.minAreaRect(points) for points in points_list]
    mr_size = np.array([mr[1] for mr in minrects])
    if len(mr_size) == 0:
        return []
    mr_size.sort(axis=1)
    p_size = np.array([len(points) for points in points_list])
    good = (p_size >= pmin) & (p_size < pmax) \
           & (mr_size[:, 0] >= wmin) & (mr_size[:, 0] < wmax) \
           & (mr_size[:, 1] >= hmin) & (mr_size[:, 1] < hmax) \

    good = np.flatnonzero(good)
    if roi is not None:
        inside = []
        for ii in good:
            vertices = np.int0(cv2.boxPoints(minrects[ii]))
            contained = [roi.containsPoint(qc.QPointF(*vtx), qc.Qt.OddEvenFill)
                             for vtx in vertices]
            if np.any(contained):
                inside.append(ii)
        good = inside
    # logging.debug(f'From {len(points_list)} indices fitting size conds: {good}')
    return [points_list[ii] for ii in good]


def get_bounding_poly(points_list: List[np.ndarray],
                      style: ut.OutlineStyle) -> List[np.ndarray]:
    """This returns a list of bounding-polygons of the list of points
    in `points_list`.

    Parameters
    ----------
    points_list: list
        List of point arrays masking each object.
    style: argos.utility.OutlineStyle

    Returns
    -------
    list[np.ndarray]
        If `style` is OutlineStyle.fill - the same list of points without
        doing anything.
        If OutlineStyle.contour - the list of points representing the
        contour of each entry in `points_list`.
        If OutlineStyle.minrect - the list of vertices of the minimum-rotated
        rectangles bounding each entry in `points_list`.
        If OutlineStyle.bbox - the list of vertices of the axis-aligned
        rectangles bounding each entry in `points_list`.

    This does not strictly extract bounding points, as when `style` is
    `OutlineStyle.filled`, it just returns the same set of points. Any client
    using a uniform policy of drawing a polygon with the returned points will be
    essentially filling it up.

    I had to make a binary image with the specified points set to 1 because
    that's what cv2.findContours takes.
    """
    if style == ut.OutlineStyle.fill:
        return points_list
    contours_list = []
    for points in points_list:
        # logging.debug('%r, %r', type(points), points)
        if style == ut.OutlineStyle.minrect:
            contours_list.append(
                np.int0(cv2.boxPoints(cv2.minAreaRect(points))))
            continue
        rect = np.array(cv2.boundingRect(points))
        if style == ut.OutlineStyle.bbox:
            contours_list.append(ut.rect2points(rect))
        elif style == ut.OutlineStyle.contour:
            # Create a binary image with the size of the bounding box
            binary_img = np.zeros((rect[3], rect[2]), dtype=np.uint8)
            # Turn on the pixels for corresponding points
            pos = points - rect[:2]
            binary_img[pos[:, 1], pos[:, 0]] = 1
            contours, hierarchy = cv2.findContours(binary_img,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            # convert contour pixel positions back to image space
            contours = [contour.squeeze() + rect[:2] for contour in contours]
            contours_list += contours

    return contours_list