# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-05-29 12:56 PM


import sys
import logging
import numpy as np
import cv2
from typing import Dict, List
from matplotlib import cm
from PyQt5 import (
    QtCore as qc,
    QtGui as qg,
    QtWidgets as qw
)
import sip

import argos.utility as util
from argos.constants import DrawingGeom, ColorMode
from argos.utility import cv2qimage, make_color, get_cmap_color


class FrameScene(qw.QGraphicsScene):
    sigPolygons = qc.pyqtSignal(dict)
    sigPolygonsSet = qc.pyqtSignal()
    sigArena = qc.pyqtSignal(qg.QPolygonF)

    def __init__(self, *args, **kwargs):
        super(FrameScene, self).__init__(*args, **kwargs)
        self.roi = None
        self.frameno = -1
        self.arena = None
        self.polygons = {}
        self.item_dict = {}
        self.label_dict = {}
        self._frame = None
        self.geom = DrawingGeom.arena
        self.grayscale = False  # frame will be converted to grayscale
        self.color_mode = ColorMode.single
        # self.autocolor = False
        self.colormap = 'viridis'
        self.max_colors = 100
        self.color = qg.QColor(qc.Qt.green)
        self.selected_color = qg.QColor(qc.Qt.blue)
        self.incomplete_color = qg.QColor(qc.Qt.magenta)
        self.linewidth = 2
        self.linestyle_selected = qc.Qt.DotLine
        self.snap_dist = 5
        self.incomplete_item = None
        self.points = []
        self.selected = []
        self.font = qw.QApplication.font()
        self.font.setBold(True)

    def _clearIncomplete(self):
        if self.incomplete_item is not None:
            self.removeItem(self.incomplete_item)
            del self.incomplete_item
            self.incomplete_item = None

    def clearItems(self):
        self.points = []
        self.selected = []
        self.polygons = {}
        self.item_dict = {}
        self.incomplete_item = None
        self.clear()

    def clearAll(self):
        self.clearItems()
        self._frame = None

    @qc.pyqtSlot(list)
    def setSelected(self, selected: List[int]) -> None:
        """Set list of selected items"""
        self.selected = selected
        for key in self.item_dict:
            if key in selected:
                if self.color_mode == ColorMode.auto:
                    color = qg.QColor(*make_color(key))
                elif self.color_mode == ColorMode.cmap:
                    color = qg.QColor(
                        *get_cmap_color(key % self.max_colors, self.max_colors,
                                        self.colormap))
                else:
                    color = self.selected_color
                pen = qg.QPen(color, self.linewidth,
                              style=self.linestyle_selected)
                self.item_dict[key].setPen(pen)
                self.item_dict[key].setZValue(1)
                self.label_dict[key].setDefaultTextColor(color)
                self.label_dict[key].setZValue(1)
            else:
                if self.color_mode == ColorMode.auto:
                    color = qg.QColor(*make_color(key))
                elif self.color_mode == ColorMode.cmap:
                    color = qg.QColor(
                        *get_cmap_color(key % self.max_colors, self.max_colors,
                                        self.colormap))
                else:
                    color = self.color
                color.setAlpha(64)  # make the non-selected items transparent
                pen = qg.QPen(color, self.linewidth)
                self.item_dict[key].setPen(pen)
                self.item_dict[key].setZValue(0)
                self.label_dict[key].setDefaultTextColor(color)
                self.label_dict[key].setZValue(0)

    @qc.pyqtSlot()
    def keepSelected(self):
        """Remove all items except the selected ones"""
        bad = set(self.item_dict.keys()) - set(self.selected)
        for key in bad:
            item = self.item_dict.pop(key)
            self.removeItem(item)
            del item
            label = self.label_dict.pop(key)
            self.removeItem(label)
            del label
            self.polygons.pop(key)
        self.sigPolygons.emit(self.polygons)
        self.sigPolygonsSet.emit()

    @qc.pyqtSlot()
    def removeSelected(self):
        for key in self.selected:
            if key not in self.polygons:
                continue
            item = self.item_dict.pop(key)
            self.removeItem(item)
            del item
            label = self.label_dict.pop(key)            
            self.removeItem(label)
            del label
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

    @qc.pyqtSlot(bool)
    def setGrayScale(self, grayscale: bool) -> None:
        self.grayscale = grayscale

    def setFrame(self, frame: np.ndarray) -> None:
        self._frame = cv2qimage(frame)
        if self.grayscale:
            self._frame = self._frame.convertToFormat(
                qg.QImage.Format_Grayscale8)

    def _addItem(self, item: np.ndarray) -> None:
        index = 0 if len(self.polygons) == 0 else max(self.polygons.keys()) + 1
        self.polygons[index] = item
        if self.geom == DrawingGeom.rectangle:
            item = qw.QGraphicsRectItem(*item)
        elif self.geom == DrawingGeom.polygon:
            poly = qg.QPolygonF([qc.QPointF(*p) for p in item])
            item = qw.QGraphicsPolygonItem(poly)
        if self.color_mode == ColorMode.auto:
            pen = qg.QPen(qg.QColor(*make_color(index)))
        elif self.color_mode == ColorMode.cmap:
            pen = qg.QPen(
                qg.QColor(*get_cmap_color(index % self.max_colors,
                                          self.max_colors,
                                          self.colormap)))
        else:
            pen = qg.QPen(self.color)
        pen.setWidth(self.linewidth)
        item.setPen(pen)
        self.addItem(item)
        self.item_dict[index] = item
        bbox = item.sceneBoundingRect()
        text = self.addText(str(index), self.font)
        self.label_dict[index] = text
        text.setDefaultTextColor(self.color)
        logging.debug(f'Scene bounding rect of {index}={bbox}')
        text.setPos(bbox.x(), bbox.y() - text.boundingRect().height())
        self.sigPolygons.emit(self.polygons)
        self.sigPolygonsSet.emit()

    def addIncompletePath(self, path: qg.QPainterPath) -> None:
        self._clearIncomplete()
        pen = qg.QPen(self.incomplete_color)
        pen.setWidth(self.linewidth)
        self.incomplete_item = self.addPath(path, pen)

    @qc.pyqtSlot(np.ndarray)
    def setArena(self, vertices: np.ndarray):
        logging.debug(f'Arena: {vertices}')
        self.clearItems()
        self.arena = qg.QPolygonF([qc.QPointF(*p) for p in vertices])
        self.addPolygon(self.arena)
        self.sigArena.emit(self.arena)
        self.setSceneRect(self.arena.boundingRect())

    @qc.pyqtSlot()
    def resetArena(self):
        logging.debug('Resetting arena')
        self.arena = None
        self.clearItems()
        self.invalidate(self.sceneRect())

    @qc.pyqtSlot(int)
    def setLineWidth(self, width):
        self.linewidth = width        
        for item in self.item_dict.values():
            pen = item.pen()
            pen.setWidth(width)
            item.setPen(pen)
        self.update()

    @qc.pyqtSlot(int)
    def setFontSize(self, size):
        self.font.setPointSize(size)
        for label in self.label_dict.values():
            if sip.isdeleted(label):
                print('Error: label deleted')                
            label.setFont(self.font)
            label.adjustSize()
        self.update()

    @qc.pyqtSlot(float)
    def setRelativeFontSize(self, size):
        if self._frame is not None:
            frame_width = max((self._frame.height(), self._frame.width()))
            size = int(frame_width * size / 100)
            self.font.setPixelSize(size)
            for label in self.label_dict.values():
                label.setFont(self.font)
                label.adjustSize()
            self.update()

    @qc.pyqtSlot(qg.QColor)
    def setColor(self, color: qg.QColor) -> None:
        """Color of completed rectangles"""
        self.color = color
        self.color_mode = ColorMode.single
        for key, item in self.item_dict.items():
            if key not in self.selected:
                pen = item.pen()
                pen.setColor(self.color)
                item.setPen(pen)
                self.label_dict[key].setDefaultTextColor(self.color)
        self.update()

    @qc.pyqtSlot(qg.QColor)
    def setSelectedColor(self, color: qg.QColor) -> None:
        """Color of selected rectangle"""
        self.selected_color = color
        for key in self.selected:
            item = self.item_dict[key]
            pen = item.pen()
            pen.setColor(self.selected_color)
            item.setPen(pen)
            self.label_dict[key].setDefaultTextColor(self.selected_color)

    @qc.pyqtSlot(bool)
    def setAutoColor(self, auto: bool):
        if auto:
            self.color_mode = ColorMode.auto
            self.linestyle_selected = qc.Qt.DotLine
            for key, item in self.item_dict.items():
                color = qg.QColor(*make_color(key))
                pen = item.pen()
                pen.setColor(color)
                item.setPen(pen)
                self.label_dict[key].setDefaultTextColor(color)
        else:
            self.color_mode = ColorMode.single
            self.linestyle_selected = qc.Qt.SolidLine
            for key, item in self.item_dict.items():
                pen = item.pen()
                pen.setColor(self.color)
                item.setPen(pen)
                self.label_dict[key].setDefaultTextColor(self.color)
            for key in self.selected:
                item = self.item_dict[key]
                pen = item.pen()
                pen.setColor(self.selected_color)
                item.setPen(pen)
                self.label_dict[key].setDefaultTextColor(self.selected_color)
        self.update()

    @qc.pyqtSlot(str, int)
    def setColormap(self, cmap, max_items):
        """Set a colormap `cmap` to use for getting unique color for each
        item where maximum number of items is `max_colors`"""
        if max_items < 1:
            # self.color_mode = ColorMode.single
            # # self.colormap = None
            # self.max_colors = 10
            # self.linestyle_selected = qc.Qt.SolidLine
            return
        try:
            get_cmap_color(0, max_items, cmap)
            self.colormap = cmap
            self.max_colors = max_items
            self.linestyle_selected = qc.Qt.DotLine
            self.color_mode = ColorMode.cmap
        except ValueError:
            self.max_colors = 10
            self.color_mode = ColorMode.single
            self.linestyle_selected = qc.Qt.SolidLine
            return
        for key, item in self.item_dict.items():
            color = qg.QColor(
                *get_cmap_color(key % self.max_colors, self.max_colors,
                                self.colormap))
            pen = item.pen()
            pen.setColor(color)
            item.setPen(pen)
        self.update()


    def setIncompleteColor(self, color: qg.QColor) -> None:
        """Color of rectangles being drawn"""
        self.incomplete_color = color
        pen = self.incomplete_item.pen()
        pen.setColor(color)
        self.incomplete_item.setPen(pen)    

    @qc.pyqtSlot(dict)
    def setRectangles(self, rects: Dict[int, np.ndarray]) -> None:
        """rects: a dict of id: (x, y, w, h)"""
        logging.debug(f'Received rectangles from {self.sender()}')
        logging.debug(f'Rectangles:\n{rects}')
        self.clearItems()
        self.polygons = rects
        for id_, rect in rects.items():
            if self.color_mode == ColorMode.auto:
                color = qg.QColor(*make_color(id_))
            elif self.color_mode  == ColorMode.cmap:
                color = qg.QColor(
                    *get_cmap_color(id_ % self.max_colors, self.max_colors, self.colormap))
            else:
                color = self.color
            item = self.addRect(*rect, qg.QPen(color, self.linewidth))
            self.item_dict[id_] = item
            text = self.addText(str(id_), self.font)
            self.label_dict[id_] = text
            text.setDefaultTextColor(color)
            text.setPos(rect[0], rect[1])
            self.item_dict[id_] = item
            logging.debug(f'Set {id_}: {rect}')
        if self.arena is not None:
            self.addPolygon(self.arena, qg.QPen(qc.Qt.red))
        self.sigPolygons.emit(self.polygons)
        self.sigPolygonsSet.emit()

    @qc.pyqtSlot(dict)
    def setPolygons(self, polygons: Dict[int, np.ndarray]) -> None:
        logging.debug(f'Received polygons from {self.sender()}')
        self.clearItems()
        for id_, poly in polygons.items():
            if len(poly.shape) != 2 or poly.shape[0] < 3:
                continue
            if self.color_mode == ColorMode.auto:
                color = qg.QColor(*make_color(id_))
            elif self.color_mode == ColorMode.cmap:
                color = qg.QColor(
                    *get_cmap_color(id_ % self.max_colors, self.max_colors, self.colormap))
            else:
                color = self.color
            self.polygons[id_] = poly
            logging.debug(f'Polygon {id_} poins shape: {poly.shape}')
            points = [qc.QPoint(point[0], point[1]) for point in poly]
            polygon = qg.QPolygonF(points)
            item = self.addPolygon(polygon, qg.QPen(color, self.linewidth))
            self.item_dict[id_] = item
            text = self.addText(str(id_), self.font)
            self.label_dict[id_] = text
            text.setDefaultTextColor(color)
            pos = np.mean(poly, axis=0)
            text.setPos(pos[0], pos[1])
            self.item_dict[id_] = item
            # logging.debug(f'Set {id_}: {poly}')
        if self.arena is not None:
            self.addPolygon(self.arena, qg.QPen(qc.Qt.red))
        self.sigPolygons.emit(self.polygons)
        self.sigPolygonsSet.emit()

    def keyPressEvent(self, ev: qg.QKeyEvent) -> None:
        if ev.key() == qc.Qt.Key_Escape:
            self.points = []
        super(FrameScene, self).keyPressEvent(ev)

    def mouseReleaseEvent(self, event: qw.QGraphicsSceneMouseEvent) -> None:
        """Start drawing arena"""
        logging.debug(f'Number of items {len(self.items())}')
        if event.button() == qc.Qt.RightButton:
            self.points = []
            self._clearIncomplete()
            return
        pos = event.scenePos().toPoint()
        pos = np.array((pos.x(), pos.y()), dtype=int)
        if self.geom == DrawingGeom.rectangle:
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
        elif self.geom == DrawingGeom.polygon or \
                self.geom == DrawingGeom.arena:
            if len(self.points) > 0:
                dvec = pos - self.points[0]
                if max(abs(dvec)) < self.snap_dist and \
                        len(self.points) > 2:
                    if self.geom == DrawingGeom.polygon:
                        self._addItem(np.array(self.points))
                    elif self.geom == DrawingGeom.arena:
                        self.setArena(np.array(self.points))
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
            pen = qg.QPen(self.incomplete_color, self.linewidth)
            if self.geom == DrawingGeom.rectangle:
                if self.incomplete_item is not None:
                    # logging.debug('AAAAA %r', len(self.items()))
                    self._clearIncomplete()
                    # logging.debug('BBBBB %r', len(self.items()))
                # logging.debug(f'BBBB points: {self.points}, pos: {pos}')
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

    def keyPressEvent(self, event: qg.QKeyEvent):
        if event.key() == qc.Qt.Key_Escape:
            self.points = []
            self._clearIncomplete()
            event.accept()

    def drawBackground(self, painter: qg.QPainter, rect: qc.QRectF) -> None:
        if self._frame is None:
            return
        if self.arena is None or len(
                self.arena) == 0:  # When we have reset the arena - and going to open another video
            arena = qc.QRectF(0, 0, self._frame.width(), self._frame.height())
            self.setSceneRect(arena)
        else:
            arena = self.arena.boundingRect()
        # logging.debug(f'arena: {arena}, {self.arena}, param: {rect}')
        if self._frame is not None:
            painter.drawImage(arena, self._frame, arena)


class FrameView(qw.QGraphicsView):
    sigSetColor = qc.pyqtSignal(qg.QColor)
    sigSetColormap = qc.pyqtSignal(str, int)
    sigSetRectangles = qc.pyqtSignal(dict)
    sigSetPolygons = qc.pyqtSignal(dict)
    sigPolygons = qc.pyqtSignal(dict)
    sigPolygonsSet = qc.pyqtSignal()
    sigViewportAreaChanged = qc.pyqtSignal(qc.QRectF)
    sigArena = qc.pyqtSignal(qg.QPolygonF)
    setArenaMode = qc.pyqtSignal()
    setRoiRectMode = qc.pyqtSignal()
    setRoiPolygonMode = qc.pyqtSignal()
    sigLineWidth = qc.pyqtSignal(int)
    sigFontSize = qc.pyqtSignal(int)
    sigRelativeFontSize = qc.pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super(FrameView, self).__init__(*args, **kwargs)
        self._makeScene()
        self.sigSetColormap.connect(self.frame_scene.setColormap)
        self.sigSetRectangles.connect(self.frame_scene.setRectangles)
        self.sigSetPolygons.connect(self.frame_scene.setPolygons)
        self.sigLineWidth.connect(self.frame_scene.setLineWidth)
        self.sigFontSize.connect(self.frame_scene.setFontSize)
        self.sigRelativeFontSize.connect(self.frame_scene.setRelativeFontSize)
        self.frame_scene.sigPolygons.connect(self.sigPolygons)
        self.frame_scene.sigPolygonsSet.connect(self.polygonsSet)
        self.frame_scene.sigArena.connect(self.sigArena)
        self.setMouseTracking(True)
        self.resetArenaAction = qw.QAction('Reset arena')
        self.resetArenaAction.triggered.connect(self.frame_scene.resetArena)
        self.zoomInAction = qw.QAction('Zoom in')
        self.zoomInAction.triggered.connect(self.zoomIn)
        self.zoomOutAction = qw.QAction('Zoom out')
        self.zoomOutAction.triggered.connect(self.zoomOut)
        self.showGrayscaleAction = qw.QAction('Show in grayscale')
        self.showGrayscaleAction.setCheckable(True)
        self.showGrayscaleAction.triggered.connect(self.frame_scene.setGrayScale)
        self.setColorAction = qw.QAction('Set color')
        self.setColorAction.triggered.connect(self.chooseColor)
        self.autoColorAction = qw.QAction('Autocolor')
        self.autoColorAction.setCheckable(True)
        self.autoColorAction.triggered.connect(self.setAutoColor)
        self.autoColorAction.triggered.connect(self.frame_scene.setAutoColor)
        self.colormapAction = qw.QAction('Colormap')
        self.colormapAction.triggered.connect(self.setColormap)
        self.colormapAction.setCheckable(True)
        self.lineWidthAction = qw.QAction('Line width')
        self.lineWidthAction.triggered.connect(self.setLW)
        self.fontSizeAction = qw.QAction('Set font size in points')        
        self.fontSizeAction.triggered.connect(self.setFontSize)
        self.relativeFontSizeAction = qw.QAction('Set font size as % of larger side of image')        
        self.relativeFontSizeAction.triggered.connect(self.setRelativeFontSize)
        self.sigSetColor.connect(self.frame_scene.setColor)
        self.setArenaMode.connect(self.frame_scene.setArenaMode)
        self.setRoiRectMode.connect(self.frame_scene.setRoiRectMode)
        self.setRoiPolygonMode.connect(self.frame_scene.setRoiPolygonMode)

    def _makeScene(self):
        """Keep this separate so that subclasses can override frame_scene with s
        ubclass of FrameScene"""
        self.frame_scene = FrameScene()
        self.setScene(self.frame_scene)

    def clearAll(self):
        self.frame_scene.clearAll()
        # self.setFrame(np.zeros((4, 4)), 0)
        self.viewport().update()

    @qc.pyqtSlot(np.ndarray, int)
    def setFrame(self, frame: np.ndarray, pos: int):
        logging.debug(f'Frame set {pos}')
        self.frame_scene.setFrame(frame)
        self.frame_scene.frameno = pos
        self.viewport().update()

    @qc.pyqtSlot()
    def setLW(self) -> None:
        input_, accept = qw.QInputDialog.getInt(
            self, 'Line-width of outline',
            'pixels',
            self.frame_scene.linewidth, min=0, max=10)
        if accept:
            self.sigLineWidth.emit(input_)

    @qc.pyqtSlot()
    def setFontSize(self) -> None:
        input_, accept = qw.QInputDialog.getInt(
            self, 'Font size',
            'points',
            self.frame_scene.font.pointSize(), min=1, max=100)
        if accept:
            self.sigFontSize.emit(input_)
        
    @qc.pyqtSlot()
    def setRelativeFontSize(self) -> None:
        input_, accept = qw.QInputDialog.getDouble(
            self, 'Font size relative to image size',
            '%',
            1, min=0.1, max=10)
        if accept:
            self.sigRelativeFontSize.emit(input_)

    @qc.pyqtSlot(bool)
    def setColormap(self, checked):
        if not checked:
            self.sigSetColormap.emit(None, 0)
            return
        input_, accept = qw.QInputDialog.getItem(self, 'Select colormap',
                                                'Colormap',
                                                ['jet',
                                                 'viridis',
                                                 'rainbow',
                                                 'autumn',
                                                 'summer',
                                                 'winter',
                                                 'spring',
                                                 'cool',
                                                 'hot',
                                                 'None'])
        logging.debug(f'Setting colormap to {input_}')
        if (input_ == 'None') or (not accept):
            self.colormapAction.setChecked(False)
            return
        max_colors, accept = qw.QInputDialog.getInt(self, 'Number of colors',
                                                    'Number of colors', 10, 1,
                                                    20)
        self.autoColorAction.setChecked(False)
        self.colormapAction.setChecked(True)
        self.sigSetColormap.emit(input_, max_colors)

    @qc.pyqtSlot()
    def chooseColor(self):
        color = qw.QColorDialog.getColor(initial=self.frame_scene.color,
                                         parent=self)
        self.sigSetColor.emit(color)
        self.colormapAction.setChecked(False)
        self.autoColorAction.setChecked(False)

    @qc.pyqtSlot(bool)
    def setAutoColor(self, checked):
        if checked:
            self.colormapAction.setChecked(False)

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
        # logging.debug(f'polygons: {poly}')
        self.sigSetPolygons.emit(poly)

    @qc.pyqtSlot()
    def polygonsSet(self):
        self.viewport().update()
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
            super(FrameView, self).wheelEvent(a0)


def test_display():
    util.init()
    app = qw.QApplication(sys.argv)
    view = FrameView()
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
