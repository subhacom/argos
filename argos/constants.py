# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-17 2:24 PM
import enum
from collections import namedtuple


class OutlineStyle(enum.Enum):
    """Enumeration for outline styles"""
    bbox = enum.auto()
    minrect = enum.auto()
    contour = enum.auto()
    fill = enum.auto()


class SegmentationMethod(enum.Enum):
    threshold = enum.auto()
    contour = enum.auto()
    dbscan = enum.auto()
    watershed = enum.auto()


class SegStep(enum.Enum):
    """Intermediate result for classical segmentation"""
    blur = enum.auto()
    threshold = enum.auto()
    segmented = enum.auto()
    filtered = enum.auto()
    final = enum.auto()


class DistanceMetric(enum.Enum):
    """Enumeration for distance metrics"""
    iou = enum.auto()
    euclidean = enum.auto()
    ios = enum.auto()   # intersection over smaller


class TrackState(enum.Enum):
    tentative = enum.auto()
    confirmed = enum.auto()
    deleted = enum.auto()


class DrawingGeom(enum.Enum):
    rectangle = enum.auto()
    polygon = enum.auto()
    arena = enum.auto()

class ColorMode(enum.Enum):
    single = enum.auto()
    cmap = enum.auto()
    auto = enum.auto()

# A change is defined by frame:int, change: int - change code, orig: int - original trackid, new: int - new trackid, idx:int index of change within same frame
# idx allows maintaining sequence of changes defined within same frame
Change = namedtuple('Change', ['frame', 'change', 'orig', 'new', 'idx'])

class ChangeCode(enum.Enum):
    """Code for user defined track changes"""
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
    ChangeCode.op_delete_cur: 'delete at current'}
