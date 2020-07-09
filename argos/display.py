# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-05-29 12:56 PM


import sys
import enum
import logging
import numpy as np
import cv2
from typing import Dict, List
from PyQt5 import (
    QtCore as qc,
    QtGui as qg,
    QtWidgets as qw
)

import argos.utility as util
from argos.utility import cv2qimage


class DrawingGeom(enum.Enum):
    rectangle = enum.auto()
    polygon = enum.auto()
    arena = enum.auto()


class Scene(qw.QGraphicsScene):
    sigPolygons = qc.pyqtSignal(dict)
    sigPolygonsSet = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(Scene, self).__init__(*args, **kwargs)
        self.arena = None
        self.polygons = {}
        self.item_dict = {}
        self.label_dict = {}
        self._frame = None
        self.geom = DrawingGeom.arena
        self.color = qg.QColor(qc.Qt.green)
        self.selected_color = qg.QColor(qc.Qt.blue)
        self.incomplete_color = qg.QColor(qc.Qt.magenta)
        self.linewidth = 2
        self.snap_dist = 5
        self.incomplete_item = None
        self.points = []
        self.selected = []

    def _clearIncomplete(self):
        if self.incomplete_item is not None:
            self.removeItem(self.incomplete_item)
            self.incomplete_item = None

    def clearItems(self):
        self.points = []
        self.selected = []
        self.polygons = {}
        self.item_dict = {}
        self.incomplete_item = None
        self.clear()

    @qc.pyqtSlot(list)
    def setSelected(self, selected: List[int]) -> None:
        """Set list of selected items"""
        self.selected = selected
        for key in self.item_dict:
            if key in selected:
                self.item_dict[key].setPen(qg.QPen(self.selected_color))
                self.label_dict[key].setDefaultTextColor(self.selected_color)
            else:
                self.item_dict[key].setPen(qg.QPen(self.color))
                self.label_dict[key].setDefaultTextColor(self.color)

    @qc.pyqtSlot()
    def keepSelected(self):
        """Remove all items except the selected ones"""
        bad = set(self.item_dict.keys()) - set(self.selected)
        for key in bad:
            self.removeItem(self.item_dict.pop(key))
            self.removeItem(self.label_dict.pop(key))
            self.polygons.pop(key)
        self.sigPolygons.emit(self.polygons)
        self.sigPolygonsSet.emit()

    @qc.pyqtSlot()
    def removeSelected(self):
        for key in self.selected:
            self.removeItem(self.item_dict.pop(key))
            self.removeItem(self.label_dict.pop(key))
            self.polygons.pop(key)
        self.selected = []
        self.sigPolygons.emit(self.polygons)
        self.sigPolygonsSet.emit()

    @qc.pyqtSlot(DrawingGeom)
    def setMode(self, mode: DrawingGeom) -> None:
        self.geom = mode

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
            self.setArena(np.array((0, 0, frame.shape[1], frame.shape[0])))
        self._frame = cv2qimage(frame)
        # self.clear()
        logging.debug(f'Diagonal sum of image: {frame.diagonal().sum()}')

    def _addItem(self, item: np.ndarray) -> None:
        index = 0 if len(self.polygons) == 0 else max(self.polygons.keys()) + 1
        self.polygons[index] = item
        if self.geom == DrawingGeom.rectangle:
            item = qw.QGraphicsRectItem(*item)
        elif self.geom == DrawingGeom.polygon:
            poly = qg.QPolygonF([qc.QPointF(*p) for p in item])
            item = qw.QGraphicsPolygonItem(poly)
        pen = qg.QPen(self.color)
        pen.setWidth(self.linewidth)
        item.setPen(pen)
        self.addItem(item)
        self.item_dict[index] = item
        bbox = item.sceneBoundingRect()
        text = self.addText(str(index))
        self.label_dict[index] = text
        text.setDefaultTextColor(self.color)
        logging.debug(f'Scene bounding rect of {index}={bbox}')
        text.setPos(bbox.x(), bbox.y())
        self.sigPolygons.emit(self.polygons)
        self.sigPolygonsSet.emit()

    def addIncompletePath(self, path: qg.QPainterPath) -> None:
        self._clearIncomplete()
        pen = qg.QPen(self.incomplete_color)
        pen.setWidth(self.linewidth)
        self.incomplete_item = self.addPath(path, pen)

    @qc.pyqtSlot(np.ndarray)
    def setArena(self, rect: np.ndarray):
        logging.debug(f'Arena: {rect}')
        # Current selection is relative to scene coordinates
        # make it relative to the original _image
        # if self.arena is not None:
        #     rect = qc.QRect(self.arena.x() + rect.x(),
        #                     self.arena.y() + rect.y(),
        #                     rect.width(),
        #                     rect.height())
        self.clearItems()
        self.arena = rect
        self.setSceneRect(qc.QRectF(*rect))

    @qc.pyqtSlot()
    def resetArena(self):
        logging.debug('Resetting arena')
        self.arena = None
        self.clearItems()
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
    def setRectangles(self, rects: Dict[int, np.ndarray]) -> None:
        """rects: a dict of id: (x, y, w, h)"""
        logging.debug(f'Received rectangles from {self.sender}')
        logging.debug(f'Rectangles:\n{rects}')
        self.clearItems()
        self.polygons = rects
        for id_, rect in rects.items():
            item = self.addRect(*rect, qg.QPen(self.color))
            self.item_dict[id_] = item
            text = self.addText(str(id_))
            self.label_dict[id_] = text
            text.setDefaultTextColor(self.color)
            text.setPos(rect[0], rect[1])
            self.item_dict[id_] = item
            logging.debug(f'Set {id_}: {rect}')
        self.sigPolygons.emit(self.polygons)
        self.sigPolygonsSet.emit()

    @qc.pyqtSlot(dict)
    def setPolygons(self, polygons: Dict[int, np.ndarray]) -> None:
        logging.debug(f'Received polygons from {self.sender()}')
        self.clearItems()
        for id_, poly in polygons.items():
            if len(poly.shape) != 2 or poly.shape[0] < 3:
                continue
            self.polygons[id_] = poly
            logging.debug(f'Polygon {id_} poins shape: {poly.shape}')
            points = [qc.QPoint(point[0], point[1]) for point in poly]
            polygon = qg.QPolygonF(points)
            item = self.addPolygon(polygon, qg.QPen(self.color))
            self.item_dict[id_] = item
            text = self.addText(str(id_))
            self.label_dict[id_] = text
            text.setDefaultTextColor(self.color)
            pos = np.mean(poly, axis=0)
            text.setPos(pos[0], pos[1])
            self.item_dict[id_] = item
            logging.debug(f'Set {id_}: {poly}')
        self.sigPolygons.emit(self.polygons)
        self.sigPolygonsSet.emit()

    def keyPressEvent(self, ev: qg.QKeyEvent) -> None:
        if ev.key() == qc.Qt.Key_Escape:
            self.points = []
        super(Scene, self).keyPressEvent(ev)

    def mouseReleaseEvent(self, event: qw.QGraphicsSceneMouseEvent) -> None:
        """Start drawing arena"""
        logging.debug(f'AAAA Number of items {len(self.items())}')
        if event.button() == qc.Qt.RightButton:
            self.points = []
            self._clearIncomplete()
            return
        pos = event.scenePos().toPoint()
        pos = np.array((pos.x(), pos.y()), dtype=int)
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
                    self._addItem(rect)
                    # logging.debug('FFFF %r', len(self.items()))
            else:
                self.points = [pos]
                logging.debug(f'XXXX Number of items {len(self.items())}\n'
                              f'pos: {pos}')
                return
        elif self.geom == DrawingGeom.polygon:
            if len(self.points) > 0:
                dvec = pos - self.points[0]
                if max(abs(dvec)) < self.snap_dist and \
                        len(self.points) > 2:
                    self._addItem(np.array(self.points))
                    self._clearIncomplete()
                    self.points = []
                    logging.debug(f'YYYY Number of items {len(self.items())}')
                    return
            self.points.append(pos)
            path = qg.QPainterPath(qc.QPointF(*self.points[0]))
            for point in self.points[1:]:
                path.lineTo(qc.QPointF(*point))
            self.addIncompletePath(path)
        else:
            raise NotImplementedError(
                f'Drawing geometry {self.geom} not implemented')
        logging.debug(f'ZZZZ Number of items {len(self.items())}')

    def mouseMoveEvent(self, event: qw.QGraphicsSceneMouseEvent) -> None:
        pos = event.scenePos()
        pos = np.array((pos.x(), pos.y()), dtype=int)
        if len(self.points) > 0:
            pen = qg.QPen(self.incomplete_color)
            pen.setWidth(self.linewidth)
            if self.geom == DrawingGeom.rectangle or self.geom == DrawingGeom.arena:
                if self.incomplete_item is not None:
                    # logging.debug('AAAAA %r', len(self.items()))
                    self._clearIncomplete()
                    # logging.debug('BBBBB %r', len(self.items()))
                logging.debug(f'BBBB points: {self.points}, pos: {pos}')
                rect = util.points2rect(self.points[-1], pos)
                self.incomplete_item = self.addRect(*rect, pen)
                # logging.debug('CCCC %r', len(self.items()))
            else:
                self._clearIncomplete()
                path = qg.QPainterPath(
                    qc.QPointF(self.points[0][0], self.points[0][1]))
                [path.lineTo(qc.QPointF(p[0], p[1])) for p in self.points[1:]]
                path.lineTo(qc.QPointF(pos[0], pos[1]))
                self.addIncompletePath(path)

    def drawBackground(self, painter: qg.QPainter, rect: qc.QRectF) -> None:
        if self._frame is None:
            return
        if self.arena is None or len(self.arena) == 0: # When we have reset the arena - and going to open another video
            arena = qc.QRectF(0, 0, self._frame.width(), self._frame.height())
            self.setSceneRect(arena)
        else:
            arena = qc.QRectF(*self.arena)
        # logging.debug(f'arena: {arena}, {self.arena}, param: {rect}')
        painter.drawImage(arena, self._frame, arena)


class Display(qw.QGraphicsView):
    sigSetRectangles = qc.pyqtSignal(dict)
    sigSetPolygons = qc.pyqtSignal(dict)
    sigPolygons = qc.pyqtSignal(dict)
    sigPolygonsSet = qc.pyqtSignal()
    sigViewportAreaChanged = qc.pyqtSignal(qc.QRectF)

    def __init__(self, *args, **kwargs):
        super(Display, self).__init__(*args, **kwargs)
        scene = Scene()
        self._framenum = 0
        self.setScene(scene)
        self.sigSetRectangles.connect(scene.setRectangles)
        self.sigSetPolygons.connect(scene.setPolygons)
        scene.sigPolygons.connect(self.sigPolygons)
        # scene.sigPolygonsSet.connect(self.sigPolygonsSet)
        scene.sigPolygonsSet.connect(self.polygonsSet)
        self.setMouseTracking(True)
        self.resetArenaAction = qw.QAction('Reset arena')
        self.resetArenaAction.triggered.connect(scene.resetArena)
        self.zoomInAction = qw.QAction('Zoom in')
        self.zoomInAction.triggered.connect(self.zoomIn)
        self.zoomOutAction = qw.QAction('Zoom out')
        self.zoomOutAction.triggered.connect(self.zoomOut)

    def clearAll(self):
        self.scene().clearAll()

    @qc.pyqtSlot(np.ndarray, int)
    def setFrame(self, frame: np.ndarray, pos: int):
        logging.debug(f'Frame set {pos}')
        self.scene().setFrame(frame)
        self._framenum = pos
        self.viewport().update()

    @qc.pyqtSlot()
    def zoomIn(self):
        self.scale(1.2, 1.2)
        rect = self.mapToScene(self.viewport().rect())
        rect = rect.boundingRect()
        self.sigViewportAreaChanged.emit(rect)

    @qc.pyqtSlot()
    def zoomOut(self):
        self.scale(1 / 1.2, 1 / 1.2)
        rect = self.mapToScene(self.viewport().rect())
        rect = rect.boundingRect()
        self.sigViewportAreaChanged.emit(rect)

    @qc.pyqtSlot(dict, int)
    def setRectangles(self, rect: dict, pos: int) -> None:
        logging.debug(f'Received rectangles from {self.sender()}, frame {pos}')
        self.sigSetRectangles.emit(rect)

    @qc.pyqtSlot(dict, int)
    def setPolygons(self, poly: dict, pos: int) -> None:
        logging.debug(f'Received polygons from {self.sender()}, frame {pos}')
        logging.debug(f'polygons: {poly}')
        self.sigSetPolygons.emit(poly)

    @qc.pyqtSlot()
    def polygonsSet(self):
        self.update()
        self.sigPolygonsSet.emit()

    def wheelEvent(self, a0: qg.QWheelEvent) -> None:
        """Zoom in or out when Ctrl+MouseWheel is used.

        Most mice send rotation in units of 1/8 of a degree and each notch goes
        15 degrees. I am changing the zoom as a
        """
        if a0.modifiers() == qc.Qt.ControlModifier:
            ndegrees = a0.angleDelta().y() / 8.0
            if ndegrees > 0:
                [self.zoomIn() for ii in range(int(ndegrees / 15))]
            else:
                [self.zoomOut() for ii in range(int(-ndegrees / 15))]
            # logging.debug('Angle %f degrees', a0.angleDelta().y() / 8)
        else:
            super(Display, self).wheelEvent(a0)


def test_display():
    util.init()
    app = qw.QApplication(sys.argv)
    view = Display()
    image = cv2.imread(
        'C:/Users/raysu/analysis/animal_tracking/bugtracking/training_images/'
        'prefix_1500.png')
    view.setFrame(image, 0)
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
