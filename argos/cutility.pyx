cimport cython
import numpy as np

import enum

cimport numpy as np

from argos.constants import OutlineStyle, DistanceMetric


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray points2rect(p0: np.ndarray, p1: np.ndarray):
    """Convert diagonally opposite vertices into (x, y, w, h) format
    rectangle.

    Returns
    -------
    np.array:
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int_t, ndim=2] rect2points(
    np.ndarray[np.int_t, ndim=1] rect):
    """Convert topleft, width, height format rectangle into four anti-clockwise
    vertices"""
    return np.vstack([rect[:2],
                      (rect[0], rect[1] + rect[3]),
                      rect[:2] + rect[2:],
                      (rect[0] + rect[2], rect[1])])


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[float_t, ndim=1] tlwh2xyrh(np.ndarray[np.int_t, ndim=1] rect):
    """Convert rectangle in top-left, width, height format into center, aspect ratio, height"""
    cdef np.ndarray ret = np.asanyarray(rect, dtype=float)
    ret[0] += ret[2] * 0.5
    ret[1] += ret[3] * 0.5
    # ret[:2] += ret[2:] * 0.5
    ret[2] /= ret[3]
    return ret


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[np.int_t, ndim=1] xyrh2tlwh(np.ndarray[float_t, ndim=1] rect):
    """Convert centre, aspect ratio, height into top-left, width, height
    format"""
    cdef float w = rect[2] * rect[3]
    cdef np.ndarray ret = np.asanyarray((round(rect[0] - w / 2.0),
                                         round(rect[1] - rect[3] / 2.0),
                                         round(w), round(rect[3])),
                                        dtype=int)
    return ret


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[np.int_t, ndim=1] rect_intersection(np.ndarray[np.int_t, ndim=1] ra, np.ndarray[np.int_t, ndim=1] rb):
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
    cdef int x, y, dx, dy
    cdef np.ndarray result = np.zeros((4,), dtype=np.int)
    x = int(max(ra[0], rb[0]))
    y = int(max(ra[1], rb[1]))
    dx = int(min(ra[0] + ra[2], rb[0] + rb[2]) - x)
    dy = int(min(ra[1] + ra[3], rb[1] + rb[3]) - y)
    if (dx > 0) and (dy > 0):
        result[0] = x
        result[1] = y
        result[2] = dx
        result[3] = dy
    return result


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef float rect_iou(np.ndarray[np.int_t, ndim=1] ra, np.ndarray[np.int_t, ndim=1] rb):
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
    cdef np.ndarray inter = rect_intersection(ra, rb)
    cdef float area_i, area_u
    cdef float ret
    area_i = inter[2] * inter[3]
    area_u = ra[2] * ra[3] + rb[2] * rb[3] - area_i
    if area_u <= 0 or area_i < 0:
        raise ValueError('Area not positive')
    ret = 1.0 * area_i / area_u
    if np.isinf(ret) or np.isnan(ret) or ret < 0:
        raise ValueError('Invalid intersection')
    return ret


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef float rect_ios(np.ndarray[np.int_t, ndim=1] ra, np.ndarray[np.int_t, ndim=1] rb):
    """Compute intersection over area of smaller of two axis-aligned
    rectangles.

    This is the ratio of the are of intersection to the area of the smaller
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
    cdef np.ndarray inter = rect_intersection(ra, rb)
    cdef float area_i = inter[2] *inter[3]
    cdef float area_a = ra[2] * ra[3]
    cdef float area_b = rb[2] * rb[3]
    if area_i < 0 or area_a <= 0 or area_b <= 0:
        raise ValueError('Area not positive')
    cdef float ret = area_i / min(area_a, area_b)
    if np.isinf(ret) or np.isnan(ret) or ret < 0:
        raise ValueError('Invalid intersection')
    return ret


# @cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[float_t, ndim=2] pairwise_distance(np.ndarray[np.int_t, ndim=2] new_bboxes, np.ndarray[np.int_t, ndim=2] bboxes,
                      object boxtype, object metric):
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
    cdef np.ndarray dist = np.zeros((new_bboxes.shape[0], bboxes.shape[0]),
                                    dtype=float)
    cdef np.ndarray centers = np.zeros((bboxes.shape[0], 2), dtype=float)
    cdef np.ndarray new_centers = np.zeros((new_bboxes.shape[0], 2), dtype=float)
    if metric == DistanceMetric.euclidean:
        centers[:, :] = bboxes[:, :2] + bboxes[:, 2:] * 0.5
        new_centers = new_bboxes[:, :2] + new_bboxes[:, 2:] * 0.5
        for ii in range(new_bboxes.shape[0]):
            for jj in range(bboxes.shape[0]):
                dist[ii, jj] = np.sum((new_centers[ii] - centers[jj]) ** 2)
    elif metric == DistanceMetric.iou:
        if boxtype == OutlineStyle.bbox:  # This can be handled efficiently
            for ii in range(new_bboxes.shape[0]):
                for jj in range(bboxes.shape[0]):
                    dist[ii, jj] = 1.0 - rect_iou(bboxes[jj], new_bboxes[ii])
        else:
            raise NotImplementedError(
                'Only handling axis-aligned bounding boxes')
    elif metric == DistanceMetric.ios:
        if boxtype == OutlineStyle.bbox:
            for ii in range(new_bboxes.shape[0]): 
                for jj in range(bboxes.shape[0]):
                    dist[ii, jj] = 1.0 - rect_ios(bboxes[jj], new_bboxes[ii])
        else:
            raise NotImplementedError(
                'Only handling axis-aligned bounding boxes')
    else:
        raise NotImplementedError(f'Unknown metric {metric}')
    return dist
