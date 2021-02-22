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


settings = util.init()


class FrameScene(qw.QGraphicsScene):
    sigPolygons = qc.pyqtSignal(dict)
    sigPolygonsSet = qc.pyqtSignal()
    sigArena = qc.pyqtSignal(qg.QPolygonF)
    sigFontSizePixels = qc.pyqtSignal(int)
    sigMousePos = qc.pyqtSignal(qc.QPointF)

    def __init__(self, *args, **kwargs):
        super(FrameScene, self).__init__(*args, **kwargs)
        self.roi = None
        self.frameno = -1
        self.arena = None
        self.polygons = {}
        self.itemDict = {}
        self.labelDict = {}
        self._frame = None
        self.geom = DrawingGeom.arena
        self.grayscale = False  # frame will be converted to grayscale
        self.colorMode = settings.value('argos/color_mode', 0)
        if self.colorMode == 0:
            self.colorMode = ColorMode.single
        elif self.colorMode == 1:
            self.colorMode = ColorMode.auto
        else:
            self.colorMode = ColorMode.cmap
        # self.autocolor = False
        self.colormap = settings.value('argos/colormap', 'viridis')
        self.maxColors = 100
        self.color = settings.value('argos/color', '#00ff00')
        self.color = qg.QColor(self.color)
        self.selectedColor = settings.value('argos/selected_color', '#0000ff')
        self.selectedColor = qg.QColor(self.selectedColor)
        self.incompleteColor = settings.value('argos/incomplete_color',
                                               '#ff00ff')
        self.incompleteColor = qg.QColor(self.incompleteColor)
        self.linewidth = settings.value('argos/linewidth', 2.0, type=float)
        self.labelInside = settings.value('argos/labelinside', True, type=bool)
        self.alphaUnselected = settings.value('argos/alpha_unselected', 255,
                                            type=int)
        # self.linestyle_selected = qc.Qt.DotLine
        self.showBbox = True
        self.showId = True
        self.snap_dist = 5
        self.incomplete_item = None
        self.points = []
        self.selected = []
        self.font = qw.QApplication.font()
        self.boldFont = qg.QFont(self.font)
        self.boldFont.setWeight(qg.QFont.Bold)
        self.textIgnoresTransformation = True
        # self.font.setBold(True)

    def _clearIncomplete(self):
        if self.incomplete_item is not None:
            self.removeItem(self.incomplete_item)
            del self.incomplete_item
            self.incomplete_item = None

    def clearItems(self):
        self.points = []
        # self.selected = []
        self.polygons = {}
        self.itemDict = {}
        self.labelDict = {}
        self.incomplete_item = None
        self.clear()

    def clearAll(self):
        self.clearItems()
        self._frame = None

    @qc.pyqtSlot(list)
    def setSelected(self, selected: List[int]) -> None:
        """Set list of selected items"""
        self.selected = selected
        if len(selected) == 0:
            self._updateItemDisplay()
            self.update()
            return
        for key in self.itemDict:
            if key in selected:
                if self.colorMode == ColorMode.auto:
                    color = qg.QColor(*make_color(key))
                elif self.colorMode == ColorMode.cmap:
                    color = qg.QColor(
                        *get_cmap_color(key % self.maxColors, self.maxColors,
                                        self.colormap))
                else:
                    color = self.selectedColor
                # Make the selected item bbox thicker
                pen = qg.QPen(color, self.linewidth + 2)
                self.itemDict[key].setPen(pen)
                self.itemDict[key].setZValue(1)
                self.labelDict[key].setDefaultTextColor(color)
                self.labelDict[key].setFont(self.boldFont)
                self.labelDict[key].setZValue(1)
            else:
                if self.colorMode == ColorMode.auto:
                    color = qg.QColor(*make_color(key))
                elif self.colorMode == ColorMode.cmap:
                    color = qg.QColor(
                        *get_cmap_color(key % self.maxColors, self.maxColors,
                                        self.colormap))
                else:
                    color = self.color
                    # make the non-selected items transparent
                color.setAlpha(self.alphaUnselected)
                pen = qg.QPen(color, self.linewidth)
                self.itemDict[key].setPen(pen)
                self.itemDict[key].setZValue(0)
                self.labelDict[key].setDefaultTextColor(color)
                self.labelDict[key].setFont(self.font)
                self.labelDict[key].setZValue(0)
        self.update()

    @qc.pyqtSlot()
    def keepSelected(self):
        """Remove all items except the selected ones"""
        bad = set(self.itemDict.keys()) - set(self.selected)
        for key in bad:
            item = self.itemDict.pop(key)
            self.removeItem(item)
            del item
            label = self.labelDict.pop(key)
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
            item = self.itemDict.pop(key)
            self.removeItem(item)
            del item
            label = self.labelDict.pop(key)
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
        if self.colorMode == ColorMode.auto:
            pen = qg.QPen(qg.QColor(*make_color(index)))
        elif self.colorMode == ColorMode.cmap:
            pen = qg.QPen(
                qg.QColor(*get_cmap_color(index % self.maxColors,
                                          self.maxColors,
                                          self.colormap)))
        else:
            pen = qg.QPen(self.color)
        pen.setWidth(self.linewidth)
        item.setPen(pen)
        self.addItem(item)
        self.itemDict[index] = item
        bbox = item.sceneBoundingRect()
        text = self.addText(str(index), self.font)
        self.labelDict[index] = text
        text.setDefaultTextColor(self.color)
        # logging.debug(f'Scene bounding rect of {index}={bbox}')
        if self.labelInside:
            text.setPos(bbox.x(), bbox.y())
        else:
            text.setPos(bbox.x(), bbox.y() - text.boundingRect().height())
        text.setFlag(qw.QGraphicsItem.ItemIgnoresTransformations,
                     self.textIgnoresTransformation)
        self.sigPolygons.emit(self.polygons)
        self.sigPolygonsSet.emit()

    def addIncompletePath(self, path: qg.QPainterPath) -> None:
        self._clearIncomplete()
        pen = qg.QPen(self.incompleteColor)
        pen.setWidth(self.linewidth)
        self.incomplete_item = self.addPath(path, pen)

    # @qc.pyqtSlot(np.ndarray)
    # def setArena(self, vertices: np.ndarray):
    #     logging.debug(f'Arena: {vertices}')
    #     self.clearItems()
    #     self.arena = qg.QPolygonF([qc.QPointF(*p) for p in vertices])
    #     self.addPolygon(self.arena)
    #     self.sigArena.emit(self.arena)
    #     self.setSceneRect(self.arena.boundingRect())

    @qc.pyqtSlot(qg.QPolygonF)
    def setArena(self, poly: qg.QPolygonF):
        self.clearItems()
        self.arena = qg.QPolygonF(poly)
        self.addPolygon(self.arena)
        self.sigArena.emit(self.arena)
        self.setSceneRect(self.arena.boundingRect())
        
    @qc.pyqtSlot()
    def resetArena(self):
        logging.debug('Resetting arena')
        self.arena = None
        self.clearItems()
        self.invalidate(self.sceneRect())

    def _setLabelInside(self, labelInside):
        for key, item in self.itemDict.items():
            text = self.labelDict[key]
            bbox = item.sceneBoundingRect()
            if labelInside:
                text.setPos(bbox.x(), bbox.y())
            else:
                text.setPos(bbox.x(), bbox.y() - text.boundingRect().height())

    @qc.pyqtSlot(bool)
    def setLabelInside(self, val):
        """If True, draw the label inside bbox, otherwise, above it"""
        self.labelInside = val
        settings.setValue('argos/labelinside', val)
        self._setLabelInside(val)
        self.update()

    @qc.pyqtSlot(float)
    def setLineWidth(self, width):
        self.linewidth = width
        settings.setValue('argos/linewidth', width)
        self._updateItemDisplay()
        self.update()

    @qc.pyqtSlot(int)
    def setFontSize(self, size):
        self.font.setPointSize(size)
        self.boldFont = qg.QFont(self.font)
        self.boldFont.setBold(True)
        self._updateItemDisplay()
        self.update()

    @qc.pyqtSlot(float)
    def setRelativeFontSize(self, size):
        if self._frame is not None:
            frame_width = max((self._frame.height(), self._frame.width()))
            size = int(frame_width * size / 100)
            self.font.setPixelSize(size)
            self.boldFont = qg.QFont(self.font)
            self.boldFont.setBold(True)
            self.sigFontSizePixels.emit(size)
            self._updateItemDisplay()
            self.update()

    @qc.pyqtSlot(int)
    def setFontSizePixels(self, size):
        self.font.setPixelSize(size)
        self.boldFont = qg.QFont(self.font)
        self.boldFont.setBold(True)
        self._updateItemDisplay()
        self.update()
    

    @qc.pyqtSlot(qg.QColor)
    def setColor(self, color: qg.QColor) -> None:
        """Color of completed rectangles"""
        self.color = color
        self.colorMode = ColorMode.single
        settings.setValue('argos/color', color.name())
        settings.setValue('argos/color_mode', 0)
        self._updateItemDisplay()
        self.update()

    @qc.pyqtSlot(qg.QColor)
    def setSelectedColor(self, color: qg.QColor) -> None:
        """Color of selected rectangle"""
        self.selectedColor = color
        settings.setValue('argos/selected_color', color.name())
        if len(self.selected) == 0:
            return
        self._updateItemDisplay()
        self.update()

    @qc.pyqtSlot(int)
    def setAlphaUnselected(self, val: int) -> None:
        assert (0 <= val) and (255 >= val), f'Alpha must be in [0, 255] range. Got {val}'
        settings.setValue('argos/alpha_unselected', val)
        self.alphaUnselected = val
        # If nothing is selected don't dim everything
        if len(self.selected) == 0:
            return
        self._updateItemDisplay()
        self.update()


    def _updateItemDisplay(self):
        """Make the selected item's bbox thicker and label font bold.

        If nothing is selected make all labels bold and normal color.

        If anything is selected, make them opaque and thicker, labels in bold
        and those unselected will have transparency from `self.alphaUnselected`
        value and labels in normal font.
        """

        unselected = set(list(self.itemDict.keys())) - set(self.selected)

        if len(self.selected) == 0:
            for key, item in self.itemDict.items():
                pen = item.pen()
                pen.setWidth(self.linewidth)
                if self.colorMode == ColorMode.single:
                    color = qg.QColor(self.color)
                else:
                    color = pen.color()
                color.setAlpha(255)
                pen.setColor(color)
                item.setPen(pen)
                label = self.labelDict[key]
                label.setDefaultTextColor(color)
                label.setFont(self.boldFont)
                label.adjustSize()
                if not self.showBbox:
                    item.setOpacity(0)
                if not self.showId:
                    label.setOpacity(0)
            return

        for key in unselected:
            item = self.itemDict[key]
            pen = item.pen()
            pen.setWidth(self.linewidth)
            if self.colorMode == ColorMode.single:
                color = qg.QColor(self.color)
            else:
                color = pen.color()
            color.setAlpha(self.alphaUnselected)
            pen.setColor(color)
            item.setPen(pen)
            self.labelDict[key].setDefaultTextColor(color)
            self.labelDict[key].setFont(self.font)
            self.labelDict[key].adjustSize()
            item.setZValue(0)

        for key in self.selected:
            if key not in self.itemDict:
                continue
            item = self.itemDict[key]
            label = self.labelDict[key]
            pen = item.pen()
            pen.setWidth(self.linewidth + 2)
            if self.colorMode == ColorMode.single:
                color = qg.QColor(self.selectedColor)
            else:
                color = pen.color()
            color.setAlpha(255)
            pen.setColor(color)
            item.setPen(pen)
            label.setDefaultTextColor(color)
            label.setFont(self.boldFont)
            label.adjustSize()
            item.setZValue(1)

        if not self.showBbox:
            [item.setOpacity(0) for item in self.itemDict.values()]
        if not self.showId:
            [label.setOpacity(0) for label in self.labelDict.values()]

    @qc.pyqtSlot(bool)
    def setAutoColor(self, auto: bool):
        if auto:
            self.colorMode = ColorMode.auto
            settings.setValue('argos/color_mode', 1)
            # self.linestyle_selected = qc.Qt.DotLine
            for key, item in self.itemDict.items():
                color = qg.QColor(*make_color(key))
                pen = item.pen()
                pen.setColor(color)
                if key in self.selected:
                    pen.setWidth(self.linewidth + 2)
                item.setPen(pen)
                # self.labelDict[key].setDefaultTextColor # this is done in _updateItemDisplay
        else:
            self.colorMode = ColorMode.single
            settings.setValue('argos/color_mode', 0)
        self._updateItemDisplay()
        self.update()

    @qc.pyqtSlot(str, int)
    def setColormap(self, cmap, max_items):
        """Set a colormap `cmap` to use for getting unique color for each
        item where maximum number of items is `maxColors`"""
        if max_items < 1:
            return
        try:
            get_cmap_color(0, max_items, cmap)
            self.colormap = cmap
            self.maxColors = max_items
            # self.linestyle_selected = qc.Qt.DotLine
            self.colorMode = ColorMode.cmap
            settings.setValue('argos/color_mode', 2)
            settings.setValue('argos/colormap', cmap)
        except ValueError:
            self.maxColors = 10
            self.colorMode = ColorMode.single
            settings.setValue('argos/color_mode', 0)
            # self.linestyle_selected = qc.Qt.SolidLine
            return
        for key, item in self.itemDict.items():
            color = qg.QColor(
                *get_cmap_color(key % self.maxColors, self.maxColors,
                                self.colormap))
            pen = item.pen()
            pen.setColor(color)
            item.setPen(pen)
        self._updateItemDisplay()
        self.update()


    def setIncompleteColor(self, color: qg.QColor) -> None:
        """Color of rectangles being drawn"""
        self.incompleteColor = color
        settings.setValue('argos/incomplete_color', color.name())
        pen = self.incomplete_item.pen()
        pen.setColor(color)
        self.incomplete_item.setPen(pen)
        self.update()

    @qc.pyqtSlot(bool)
    def setShowBbox(self, val: bool) -> None:
        self.showBbox = val
        if val:
            for item in self.itemDict.values():
                item.setOpacity(1.0)
        else:
            for item in self.itemDict.values():
                item.setOpacity(0.0)
        self.update()

    @qc.pyqtSlot(bool)
    def setShowId(self, val: bool) -> None:
        self.showId = val
        if val:
            for item in self.labelDict.values():
                item.setOpacity(1.0)
        else:
            for item in self.labelDict.values():
                item.setOpacity(0.0)
        self.update()

    @qc.pyqtSlot(dict)
    def setRectangles(self, rects: Dict[int, np.ndarray]) -> None:
        """rects: a dict of id: (x, y, w, h)"""
        logging.debug(f'Received rectangles from {self.sender()}')
        logging.debug(f'Rectangles:\n{rects}')
        self.clearItems()
        self.polygons = rects
        for id_, rect in rects.items():
            if self.colorMode == ColorMode.auto:
                color = qg.QColor(*make_color(id_))
            elif self.colorMode  == ColorMode.cmap:
                color = qg.QColor(
                    *get_cmap_color(id_ % self.maxColors, self.maxColors, self.colormap))
            else:
                color = self.color
            item = self.addRect(*rect, qg.QPen(color, self.linewidth))
            self.itemDict[id_] = item
            text = self.addText(str(id_), self.boldFont)
            text.setFlag(qw.QGraphicsItem.ItemIgnoresTransformations,
                         self.textIgnoresTransformation)
            self.labelDict[id_] = text
        self._updateItemDisplay()
        self._setLabelInside(self.labelInside)
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
            if self.colorMode == ColorMode.auto:
                color = qg.QColor(*make_color(id_))
            elif self.colorMode == ColorMode.cmap:
                color = qg.QColor(
                    *get_cmap_color(id_ % self.maxColors, self.maxColors, self.colormap))
            else:
                color = self.color
            self.polygons[id_] = poly
            # logging.debug(f'Polygon {id_} points shape: {poly.shape}')
            points = [qc.QPoint(point[0], point[1]) for point in poly]
            polygon = qg.QPolygonF(points)
            item = self.addPolygon(polygon, qg.QPen(color, self.linewidth))
            self.itemDict[id_] = item
            text = self.addText(str(id_), self.font)
            self.labelDict[id_] = text
            text.setFlag(qw.QGraphicsItem.ItemIgnoresTransformations,
                         self.textIgnoresTransformation)
        self._updateItemDisplay()
        self._setLabelInside(self.labelInside)
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
                rect = qg.QPolygonF([qc.QPointF(*p) for p in rect])
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
                        poly = qg.QPolygonF([qc.QPointF(*p) for p in self.points])
                        self.setArena(poly)
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
        self.sigMousePos.emit(pos)
        pos = np.array((pos.x(), pos.y()), dtype=int)
        if len(self.points) > 0:
            pen = qg.QPen(self.incompleteColor, self.linewidth)
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
    sigSetSelectedColor = qc.pyqtSignal(qg.QColor)
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
    # sigSetLabelInside = qc.pyqtSignal(bool)
    sigLineWidth = qc.pyqtSignal(float)
    sigFontSize = qc.pyqtSignal(int)
    sigRelativeFontSize = qc.pyqtSignal(float)
    sigSetAlphaUnselected = qc.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super(FrameView, self).__init__(*args, **kwargs)
        self._makeScene()
        self.sigSetColormap.connect(self.frameScene.setColormap)
        self.sigSetAlphaUnselected.connect(self.frameScene.setAlphaUnselected)
        self.sigSetRectangles.connect(self.frameScene.setRectangles)
        self.sigSetPolygons.connect(self.frameScene.setPolygons)
        self.sigLineWidth.connect(self.frameScene.setLineWidth)
        self.sigFontSize.connect(self.frameScene.setFontSize)
        self.sigRelativeFontSize.connect(self.frameScene.setRelativeFontSize)
        self.frameScene.sigPolygons.connect(self.sigPolygons)
        self.frameScene.sigPolygonsSet.connect(self.polygonsSet)
        self.frameScene.sigArena.connect(self.sigArena)
        self.setMouseTracking(True)
        self.resetArenaAction = qw.QAction('Reset arena')
        self.resetArenaAction.triggered.connect(self.frameScene.resetArena)
        self.zoomInAction = qw.QAction('Zoom in')
        self.zoomInAction.triggered.connect(self.zoomIn)
        self.zoomOutAction = qw.QAction('Zoom out')
        self.zoomOutAction.triggered.connect(self.zoomOut)
        self.showGrayscaleAction = qw.QAction('Show in grayscale')
        self.showGrayscaleAction.setCheckable(True)
        self.showGrayscaleAction.triggered.connect(self.frameScene.setGrayScale)
        self.setColorAction = qw.QAction('Set color')
        self.setColorAction.triggered.connect(self.chooseColor)
        self.setSelectedColorAction = qw.QAction('Set color of selected ID')
        self.setSelectedColorAction.triggered.connect(self.chooseSelectedColor)
        self.setAlphaUnselectedAction = qw.QAction('Set opacity of non-selected IDs')
        self.setAlphaUnselectedAction.triggered.connect(self.setAlphaUnselected)
        self.autoColorAction = qw.QAction('Autocolor')
        self.autoColorAction.setCheckable(True)
        self.autoColorAction.triggered.connect(self.setAutoColor)
        self.autoColorAction.triggered.connect(self.frameScene.setAutoColor)
        self.colormapAction = qw.QAction('Colormap')
        self.colormapAction.triggered.connect(self.setColormap)
        self.colormapAction.setCheckable(True)
        self.setLabelInsideAction = qw.QAction('Label inside bbox')
        self.setLabelInsideAction.setCheckable(True)
        self.setLabelInsideAction.setChecked(self.frameScene.labelInside)
        self.setLabelInsideAction.triggered.connect(self.frameScene.setLabelInside)
        self.lineWidthAction = qw.QAction('Line width')
        self.lineWidthAction.triggered.connect(self.setLW)
        self.fontSizeAction = qw.QAction('Set font size in points')        
        self.fontSizeAction.triggered.connect(self.setFontSize)
        self.relativeFontSizeAction = qw.QAction('Set font size as % of larger side of image')        
        self.relativeFontSizeAction.triggered.connect(self.setRelativeFontSize)
        self.showBboxAction = qw.QAction('Show boundaries')
        self.showBboxAction.setCheckable(True)
        self.showBboxAction.setChecked(True)
        self.showBboxAction.triggered.connect(self.frameScene.setShowBbox)
        self.showIdAction = qw.QAction('Show Ids')
        self.showIdAction.setCheckable(True)
        self.showIdAction.setChecked(True)
        self.showIdAction.triggered.connect(self.frameScene.setShowId)
        self.sigSetColor.connect(self.frameScene.setColor)
        self.sigSetSelectedColor.connect(self.frameScene.setSelectedColor)
        self.setArenaMode.connect(self.frameScene.setArenaMode)
        self.setRoiRectMode.connect(self.frameScene.setRoiRectMode)
        self.setRoiPolygonMode.connect(self.frameScene.setRoiPolygonMode)

    def _makeScene(self):
        """Keep this separate so that subclasses can override frameScene with s
        ubclass of FrameScene"""
        self.frameScene = FrameScene()
        self.setScene(self.frameScene)

    def clearAll(self):
        self.frameScene.clearAll()
        # self.setFrame(np.zeros((4, 4)), 0)
        self.viewport().update()

    @qc.pyqtSlot(np.ndarray, int)
    def setFrame(self, frame: np.ndarray, pos: int):
        logging.debug(f'Frame set {pos}')
        self.frameScene.setFrame(frame)
        self.frameScene.frameno = pos
        self.viewport().update()

    @qc.pyqtSlot()
    def setLW(self) -> None:
        input_, accept = qw.QInputDialog.getDouble(
            self, 'Line-width of outline',
            'pixels',
            self.frameScene.linewidth, min=0, max=10)
        if accept:
            self.sigLineWidth.emit(input_)

    @qc.pyqtSlot()
    def setFontSize(self) -> None:
        input_, accept = qw.QInputDialog.getInt(
            self, 'Font size',
            'points',
            self.frameScene.font.pointSize(), min=1, max=100)
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
        color = qw.QColorDialog.getColor(initial=self.frameScene.color,
                                         parent=self)
        self.sigSetColor.emit(color)
        self.colormapAction.setChecked(False)
        self.autoColorAction.setChecked(False)

    @qc.pyqtSlot()
    def chooseSelectedColor(self):
        color = qw.QColorDialog.getColor(initial=self.frameScene.selectedColor,
                                         parent=self)
        self.sigSetSelectedColor.emit(color)

    @qc.pyqtSlot()
    def setAlphaUnselected(self):
        alpha, accept = qw.QInputDialog.getInt(self, 'Opacity of unselected IDs',
                                       'Alpha (0=fully transparent, 255=fully opaque)',
                                       value=self.frameScene.alphaUnselected,
                                       min=0, max=255)
        print('Received', alpha)
        if accept:
            self.sigSetAlphaUnselected.emit(alpha)

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
