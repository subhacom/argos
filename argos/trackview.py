# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2022-02-20 11:10 PM
import logging
from typing import Dict

import numpy as np
from PyQt5 import QtCore as qc, QtWidgets as qw, QtGui as qg
from argos.frameview import FrameView, FrameScene
from argos.utility import get_cmap_color
from argos import utility as ut


settings = ut.init()


class TrackView(FrameView):
    """Visualization of bboxes of objects on video frame with facility to set
    visible area of scene"""

    sigSelected = qc.pyqtSignal(list)
    sigTrackDia = qc.pyqtSignal(float)
    sigTrackMarkerThickness = qc.pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super(TrackView, self).__init__(*args, **kwargs)
        self.sigSelected.connect(self.scene().setSelected)
        self.sigTrackDia.connect(self.scene().setPathDia)
        self.sigTrackMarkerThickness.connect(
            self.frameScene.setTrackMarkerThickness
        )

    def setViewportRect(self, rect: qc.QRectF) -> None:
        self.fitInView(
            rect.x(),
            rect.y(),
            rect.width(),
            rect.height(),
            qc.Qt.KeepAspectRatio,
        )

    def _makeScene(self):
        self.frameScene = ReviewScene()
        self.setScene(self.frameScene)

    @qc.pyqtSlot()
    def setPathDia(self):
        input_, accept = qw.QInputDialog.getDouble(
            self,
            'Diameter of path markers',
            'pixels',
            self.frameScene.pathDia,
            min=0,
            max=500,
        )
        if accept:
            self.sigTrackDia.emit(input_)

    @qc.pyqtSlot()
    def setTrackMarkerThickness(self):
        input_, accept = qw.QInputDialog.getDouble(
            self,
            'Thickness of path markers',
            'pixels',
            self.frameScene.markerThickness,
            min=0,
            max=500,
        )
        if accept:
            self.sigTrackMarkerThickness.emit(input_)

    @qc.pyqtSlot(bool)
    def enableDraw(self, enable: bool):
        """Activate arena drawing"""
        self.frameScene.disableDrawing(not enable)


class ReviewScene(FrameScene):
    def __init__(self, *args, **kwargs):
        super(ReviewScene, self).__init__(*args, **kwargs)
        self.drawingDisabled = True
        self.lineStyleOldTrack = qc.Qt.DashLine
        self.histGradient = 1
        self.trackHist = []
        self.markerThickness = settings.value(
            'review/marker_thickness', 2.0, type=float
        )
        self.pathDia = settings.value('review/path_diameter', 5)
        self.trackStyle = '-'  # `-` for line, `o` for ellipse
        # TODO implement choice of the style

    @qc.pyqtSlot(int)
    def setHistGradient(self, age: int) -> None:
        self.histGradient = age

    @qc.pyqtSlot(float)
    def setTrackMarkerThickness(self, thickness: float) -> None:
        """Set the thickness of the marker-edge for drawing paths"""
        self.markerThickness = thickness
        settings.setValue('review/marker_thickness', thickness)

    @qc.pyqtSlot(np.ndarray, str)
    def showTrackHist(self, track: np.ndarray, cmap: str) -> None:
        for item in self.trackHist:
            try:
                self.removeItem(item)
            except Exception as e:
                logging.debug(f'{e}')
                pass
        self.trackHist = []
        if self._frame is None:
            return
        colors = [
            qg.QColor(*get_cmap_color(ii, len(track), cmap))
            for ii in range(len(track))
        ]
        pens = [
            qg.QPen(qg.QBrush(color), self.markerThickness) for color in colors
        ]
        for item in self.items():
            if isinstance(item, qw.QGraphicsLineItem):
                print(item)
        if self.trackStyle == '-':
            self.trackHist = [
                self.addLine(
                    track[ii - 1][0],
                    track[ii - 1][1],
                    track[ii][0],
                    track[ii][1],
                    pens[ii],
                )
                for ii in range(1, len(track))
            ]
        elif self.trackStyle == 'o':
            self.trackHist = [
                self.addEllipse(
                    track[ii][0],
                    track[ii][1],
                    self.pathDia,
                    self.pathDia,
                    pens[ii],
                )
                for ii in range(len(track))
            ]

    @qc.pyqtSlot(float)
    def setPathDia(self, val):
        self.pathDia = val

    @qc.pyqtSlot(dict)
    def setRectangles(self, rects: Dict[int, np.ndarray]) -> None:
        """rects: a dict of id: (x, y, w, h, frame)

        This overrides the same slot in FrameScene where each rectangle has
        a fifth entry indicating frame no of the rectangle.

        The ones from earlier frame that are not present in the current frame
        are displayed with a special line style (default: dashes)
        """
        logging.debug(
            f'{self.objectName()} Received rectangles from {self.sender().objectName()}'
        )
        logging.debug(f'{self.objectName()} Rectangles: {rects}')
        logging.debug(f'{self.objectName()} cleared')
        self.clearItems()
        tmpRects = {id_: rect[:4] for id_, rect in rects.items()}
        super(ReviewScene, self).setRectangles(tmpRects)

        for id_, tdata in rects.items():
            if tdata.shape[0] != 5:
                raise ValueError(f'Incorrectly sized entry: {id_}: {tdata}')
            item = self.itemDict[id_]
            label = self.labelDict[id_]
            if tdata[4] < self.frameno:
                alpha = int(
                    255
                    * (
                        1
                        - 0.9
                        * min(
                            np.abs(self.frameno - tdata[4]), self.histGradient
                        )
                        / self.histGradient
                    )
                )
                pen = item.pen()
                color = pen.color()
                color.setAlpha(alpha)
                pen.setColor(color)
                pen.setStyle(self.lineStyleOldTrack)
                item.setPen(pen)
                label.setDefaultTextColor(color)
                label.setFont(self.font)
                label.adjustSize()
        self.update()

    def clearAll(self):
        super(ReviewScene, self).clearAll()
        self.trackHist = []
        self.selected = []
