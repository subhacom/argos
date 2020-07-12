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
    dbscan = enum.auto()
    threshold = enum.auto()
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


class TrackState(enum.Enum):
    tentative = enum.auto()
    confirmed = enum.auto()
    deleted = enum.auto()


class DrawingGeom(enum.Enum):
    rectangle = enum.auto()
    polygon = enum.auto()
    arena = enum.auto()


Change = namedtuple('Change', ['frame', 'change', 'orig', 'new'])