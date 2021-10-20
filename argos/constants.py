# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-17 2:24 PM
"""
=================================
Constants defined for Argos tools
=================================
"""
import enum
from collections import namedtuple


class OutlineStyle(enum.Enum):
    """Outline styles for objects.

    Attributes
    ----------
    bbox:
        Axis-aligned bounding box.
    minrect:
        Minimum area bounding rectangle, this is a tighter
        rectangle than bbox, and can be at an angle.
    contour:
        Contour of the object.
    fill:
        Color-filled contour of the object

    """

    bbox = enum.auto()
    minrect = enum.auto()
    contour = enum.auto()
    fill = enum.auto()


class SegmentationMethod(enum.Enum):
    """Segmentation methods.

    Attributes
    ----------
    threshold:
        Use thresholding and then bounding boxes of the blobs.
    contour:
        Use thresholding and then filled-contours of the blobs.
    dbscan:
        Use thresholding and then spatially cluster the non-zero
        pixels with DBSCAN algorithm.
    watershed:
        Use watershed algorithm for segmentation.
    """

    threshold = enum.auto()
    contour = enum.auto()
    dbscan = enum.auto()
    watershed = enum.auto()


class SegStep(enum.Enum):
    """Intermediate steps when doing image segmentation using classical
    image processing methods.

    Attributes
    ----------
    blur:
       Blurring.
    threshold:
       Thresholding of blurred image.
    segmented:
        Segmentation of the blurred and thresholded image.
    filtered:
        After filtering the segmented objects based on area, width,
        height, etc.
    final:
        Final segmentation. This is a placeholder to avoid showing any
        separate window for the results of the intermediate steps.
    """

    blur = enum.auto()
    threshold = enum.auto()
    segmented = enum.auto()
    filtered = enum.auto()
    final = enum.auto()


class DistanceMetric(enum.Enum):
    """Distance metrics.

    Attributes
    ----------
    iou:
       Intersection over Union, this is a very common metric where the
       area of intersection of two patches is divided by their union,
       producing 0 for no-overlap and 1 for complete overlap.
    euclidean:
       Euclidean distance between objects (usually their centres).
    ios:
       Intersection over smaller. A slight variation of IoU to account
       for the fact that a small object may be completely overlapping
       a large object, and yet the IoU will be very small. In our
       segmentation, we want to merge such objects, and a large IoS
       suggests merge.
    """

    iou = enum.auto()
    euclidean = enum.auto()
    ios = enum.auto()  # intersection over smaller


class TrackState(enum.Enum):
    """Possible tracking states of an object."""

    tentative = enum.auto()
    confirmed = enum.auto()
    deleted = enum.auto()


class DrawingGeom(enum.Enum):
    """The kind of geometry we are drawing.

    A rectangle is defined by (x, y, w, h) but a polygon is a list of
    vertices (a rectangle could also be defined with this, but the
    drawing algorithm would be different).

    Arena is a special case, when we want to change the
    visible/processed area of an image, not just draw a polygon on it.

    Attributes
    ----------
    rectangle:
        Rectangle (use ``(x, y, w, h)`` format)

    polygon:
        Polygon defined by sequence of vertices like ``((x0, y0), (x1,
        y1), (x2, y2), ...)``

    arena:
        A special case of polygon for defining the area of interest in
        the image.

    """

    rectangle = enum.auto()
    polygon = enum.auto()
    arena = enum.auto()


class ColorMode(enum.Enum):
    """Coloring scheme to be used when drawing object boundaries and IDs.

    Attributes
    ----------
    single:
        A single color for all objects. Selected objects in a different color.
    cmap:
        Pick a set of colors from a colormap and rotate.
    auto:
        Pick a random color for each object.

    """

    single = enum.auto()
    cmap = enum.auto()
    auto = enum.auto()


Change = namedtuple('Change', ['frame', 'end', 'change', 'orig', 'new', 'idx'])
"""
A change action by the user when revieweing tracks.
It is defined by the following attributes:

Attributes
----------
frame:
   frame number on which the change was applied.
end:
   frame number till (inclusive) which the change should be applied. -1 for
   till end of the video.
change:
   change code, defined in ChangeCode enum
orig:
   original trackid
new:
   new trackid
idx:
   index of change within same frame idx allows maintaining sequence of
   changes defined within same frame.
"""


class ChangeCode(enum.Enum):
    """Code for user defined track changes.
    These are:

    In this and all future frames:

    Attributes
    ----------
    op_swap:
        Swap IDs (``new`` becomes ``orig`` and ``orig`` becomes ``new``).
    op_assign:
        Assign ``new`` ID to ``orig`` ID.
    op_delete:
        Delete ``orig`` (``new`` not required)
    op_merge:
        Merge ``new`` into ``orig``, this is kept for possible future
        extension.


    Same as above, but apply only in the current frame:

    Attributes
    ----------
    op_swap_cur
    op_assign_cur
    op_delete_cur

    """

    op_swap = enum.auto()
    op_swap_cur = enum.auto()
    op_assign = enum.auto()
    op_assign_cur = enum.auto()  # Assign trackid only for current frame
    op_delete = enum.auto()
    op_delete_cur = enum.auto()
    op_merge = enum.auto()


change_name = {
    ChangeCode.op_assign: 'assign',
    ChangeCode.op_merge: 'merge',
    ChangeCode.op_swap: 'swap',
    ChangeCode.op_delete: 'delete',
    ChangeCode.op_assign_cur: 'assign at current',
    ChangeCode.op_swap_cur: 'swap at current',
    ChangeCode.op_delete_cur: 'delete at current',
}
"""
Dict mapping the change codes to their human readable names.
"""

#: number of frames to extract by default
EXTRACT_FRAMES = 200

#: styles
STYLE_SHEETS = ['default', 'light', 'dark']
