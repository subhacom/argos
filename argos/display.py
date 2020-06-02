# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-05-29 12:56 PM


import sys
import enum
import logging
from typing import Dict
import numpy as np
import cv2
from PyQt5 import (
    QtCore as qc,
    QtGui as qg,
    QtWidgets as qw
)

import argos.utility as util


class DrawingGeom(enum.Enum):
    rectangle = enum.auto()
    polygon = enum.auto()
    arena = enum.auto()


def cv2qimage(frame: np.ndarray, copy: bool=False) -> qg.QImage:
    """Convert BGR/gray/bw frame into QImage"""
    if (len(frame.shape) == 3) and (frame.shape[2] == 3):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        qimg = qg.QImage(img.tobytes(), w, h, w * c, qg.QImage.Format_RGB888)
    elif len(frame.shape) == 2:  # grayscale
        h, w = frame.shape
        qimg = qg.QImage(frame.tobytes(), w, h, w * 1, qg.QImage.Format_Grayscale8)
    if copy:
        return qimg.copy()
    return qimg


class Scene(qw.QGraphicsScene):
    def __init__(self, *args, **kwargs):
        super(Scene, self).__init__(*args, **kwargs)
        self.arena = None
        self.polygons = {}
        self._frame = None
        self.geom = DrawingGeom.arena
        self.color = qg.QColor(qc.Qt.yellow)
        self.selected_color = qg.QColor(qc.Qt.green)
        self.incomplete_color = qg.QColor(qc.Qt.magenta)
        self.linewidth = 2
        self.snap_dist = 5
        self.incomplete_item = None
        self.points = []

    def _clearIncomplete(self):
        if self.incomplete_item is not None:
            self.removeItem(self.incomplete_item)
            self.incomplete_item = None

    def _clear(self):
        self.polygons = {}
        self.points = []
        self.incomplete_item = None
        self.clear()

    @qc.pyqtSlot()
    def setArenaMode(self):
        self.geom = DrawingGeom.arena

    @qc.pyqtSlot()
    def setRoiRectMode(self):
        self.geom = DrawingGeom.rectangle

    @qc.pyqtSlot()
    def setRoiPolygonMode(self):
        self.geom = DrawingGeom.polygon

    def setFrame(self, frame: np.ndarray) -> None:
        if self.arena is None:
            self.setArena(qc.QRect(0, 0, frame.shape[1], frame.shape[0]))
        self._frame = cv2qimage(frame)
        #self.clear()
        logging.debug(f'Diagonal sum of image: {frame.diagonal().sum()}')

    def _addItem(self, item: qw.QGraphicsItem) -> None:
        index = 0 if len(self.polygons) == 0 else max(self.polygons.keys()) + 1
        self.polygons[index] = item
        pen = qg.QPen(self.color)
        pen.setWidth(self.linewidth)
        item.setPen(pen)
        self.addItem(item)
        bbox = item.sceneBoundingRect()
        text = self.addText(str(index))
        text.setDefaultTextColor(self.color)
        logging.debug(f'Scene bounding rect of {index}={bbox}')
        text.setPos(bbox.x(), bbox.y())

    def addIncompletePath(self, path: qg.QPainterPath) -> None:
        self._clearIncomplete()
        pen = qg.QPen(self.incomplete_color)
        pen.setWidth(self.linewidth)
        self.incomplete_item = self.addPath(path, pen)

    @qc.pyqtSlot(qc.QRect)
    def setArena(self, rect: qc.QRect):
        logging.debug(f'Arena: {rect}')
        # Current selection is relative to scene coordinates
        # make it relative to the original _image
        # if self.arena is not None:
        #     rect = qc.QRect(self.arena.x() + rect.x(),
        #                     self.arena.y() + rect.y(),
        #                     rect.width(),
        #                     rect.height())
        self._clear()
        self.arena = rect
        self.setSceneRect(qc.QRectF(rect))

    @qc.pyqtSlot()
    def resetArena(self):
        self.arena = None
        self._clear()
        self.invalidate(self.sceneRect())

    def setLineWidth(self, width):
        self.linewidth = width

    def setColor(self, color: qg.QColor) -> None:
        """Color of completed rectangles"""
        self.color = color

    def setSelectedColor(self, color: qg.QColor) -> None:
        """Color of selected rectangle"""
        self.selected_color = color

    def setIncompleteColor(self, color: qg.QColor) -> None:
        """Color of rectangles being drawn"""
        self.incomplete_color = color

    @qc.pyqtSlot(dict)
    def setRectangles(self, rects: dict):
        """rects: a dict of id: (x, y, w, h)"""
        self.polygons = {}
        self._clear()
        for id_, rect in rects.items():
            item = self.addRect(*rect, qg.QPen(self.color))
            text = self.addText(str(id_))
            text.setDefaultTextColor(self.color)
            text.setPos(rect[0], rect[1])
            self.polygons[id_] = item
            logging.debug(f'Set {id_}: {rect}')

    def keyPressEvent(self, ev: qg.QKeyEvent) -> None:
        if ev.key() == qc.Qt.Key_Escape:
            self.points = []
        super(Scene, self).keyPressEvent(ev)

    def mouseReleaseEvent(self, event: qw.QGraphicsSceneMouseEvent) -> None:
        """Start drawing arena"""
        logging.debug(f'AAAA Number of items {len(self.items())}')
        if event.button() == qc.Qt.RightButton:
            self.points = []
            return
        pos = event.scenePos().toPoint()
        if self.geom == DrawingGeom.rectangle or \
                self.geom == DrawingGeom.arena:
            if len(self.points) > 0:
                rect = util.points2rect(self.points[0], pos)
                self.points = []
                if self.geom == DrawingGeom.arena:
                    self.setArena(rect)
                else:
                    # logging.debug('DDDD %r', len(self.items()))
                    self._clearIncomplete()
                    # logging.debug('EEEE %r', len(self.items()))
                    self._addItem(qw.QGraphicsRectItem(rect))
                    # logging.debug('FFFF %r', len(self.items()))
            else:
                self.points= [pos]
                logging.debug(f'XXXX Number of items {len(self.items())}')
                return
        elif self.geom == DrawingGeom.polygon:
            if len(self.points) > 0:
                dvec = pos - self.points[0]
                if dvec.manhattanLength() < self.snap_dist and \
                        len(self.points) > 2:
                    self._addItem(qw.QGraphicsPolygonItem(
                        qg.QPolygonF(self.points)))
                    self._clearIncomplete()
                    self.points = []
                    logging.debug(f'YYYY Number of items {len(self.items())}')
                    return
            self.points.append(pos)
            path = qg.QPainterPath(self.points[0])
            for point in self.points[1:]:
                path.lineTo(point)
            self.addIncompletePath(path)
        else:
            raise NotImplementedError(
                f'Drawing geometry {self.geom} not implemented')
        logging.debug(f'ZZZZ Number of items {len(self.items())}')

    def mouseMoveEvent(self, event: qw.QGraphicsSceneMouseEvent) -> None:
        pos = event.scenePos()
        if len(self.points) > 0:
            pen = qg.QPen(self.incomplete_color)
            pen.setWidth(self.linewidth)
            if self.geom == DrawingGeom.rectangle or self.geom == DrawingGeom.arena:
                if self.incomplete_item is not None:
                    # logging.debug('AAAAA %r', len(self.items()))
                    self._clearIncomplete()
                    # logging.debug('BBBBB %r', len(self.items()))
                rect = util.points2rect(self.points[-1], pos)
                self.incomplete_item = self.addRect(rect, pen)
                # logging.debug('CCCC %r', len(self.items()))
            else:
                self._clearIncomplete()
                path = qg.QPainterPath(self.points[0])
                [path.lineTo(p) for p in self.points[1:]]
                path.lineTo(pos)
                self.addIncompletePath(path)

    def drawBackground(self, painter: qg.QPainter, rect: qc.QRectF) -> None:
        if self._frame is None:
            return
        if self.arena is None: # When we have reset the arena - and going to open another video
            arena = qc.QRectF(0, 0, self._frame.width(), self._frame.height())
            self.setSceneRect(arena)
        else:
            arena = self.arena
        painter.drawImage(arena, self._frame, arena)


class Display(qw.QGraphicsView):

    sigSetRectangles = qc.pyqtSignal(dict)

    def __init__(self, *args, **kwargs):
        super(Display, self).__init__(*args, **kwargs)
        scene = Scene()
        self.setScene(scene)
        self.sigSetRectangles.connect(scene.setRectangles)
        self.setMouseTracking(True)
        self.resetArenaAction = qw.QAction('Reset arena')
        self.resetArenaAction.triggered.connect(scene.resetArena)

    @qc.pyqtSlot(np.ndarray, int)
    def setFrame(self, frame: np.ndarray, pos: int):
        logging.debug(f'Frame set {pos}')
        self.scene().setFrame(frame)
        self.viewport().update()

    @qc.pyqtSlot()
    def zoomIn(self):
        self.scale(1.2, 1.2)

    @qc.pyqtSlot()
    def zoomOut(self):
        self.scale(1/1.2, 1/1.2)


def test_display():
    util.init()
    app = qw.QApplication(sys.argv)
    view = Display()
    image = cv2.imread(
        'C:/Users/raysu/analysis/animal_tracking/bugtracking/training_images/'
        'prefix_1500.png')
    view.setFrame(image)
    win = qw.QMainWindow()
    toolbar = win.addToolBar('Zoom')
    zi = qw.QAction('Zoom in')
    zi.triggered.connect(view.zoomIn)
    zo = qw.QAction('Zoom out')
    zo.triggered.connect(view.zoomOut)
    arena = qw.QAction('Select arena')
    arena.triggered.connect(view.scene().setArenaMode)
    arena_reset = qw.QAction('Rset arena')
    arena_reset.triggered.connect(view.scene().resetArena)
    roi = qw.QAction('Select rectangular ROIs')
    roi.triggered.connect(view.scene().setRoiRectMode)
    poly = qw.QAction('Select polygon ROIs')
    poly.triggered.connect(view.scene().setRoiPolygonMode)
    toolbar.addAction(zi)
    toolbar.addAction(zo)
    toolbar.addAction(arena)
    toolbar.addAction(roi)
    toolbar.addAction(poly)
    toolbar.addAction(arena_reset)
    win.setCentralWidget(view)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    test_display()

