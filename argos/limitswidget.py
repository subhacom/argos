# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-08-11 9:20 PM
"""Widget to apply size constraints and ROI to filter segmented objects"""
from typing import List, Tuple
import logging
from collections import OrderedDict
import numpy as np
import cv2
from sklearn import cluster
from PyQt5 import (
    QtWidgets as qw,
    QtCore as qc,
    QtGui as qg
)

import argos.constants
from argos import utility as ut
from argos.frameview import FrameView


settings = ut.init()


class LimitsWidget(qw.QWidget):
    sigProcessed = qc.pyqtSignal(np.ndarray, int)
    sigQuit = qc.pyqtSignal()
    sigWmin = qc.pyqtSignal(int)
    sigWmax = qc.pyqtSignal(int)
    sigHmin = qc.pyqtSignal(int)
    sigHmax = qc.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super(LimitsWidget, self).__init__(*args, **kwargs)
        self.roi = None
        layout = qw.QFormLayout()
        self._wmin_label = qw.QLabel('Minimum width')
        self._wmin_edit = qw.QSpinBox()
        self._wmin_edit.setRange(1, 1000)
        value = settings.value('segment/min_width',10,
                               type=int)
        self._wmin_edit.setValue(value)
        layout.addRow(self._wmin_label, self._wmin_edit)
        self._wmax_label = qw.QLabel('Maximum width')
        self._wmax_edit = qw.QSpinBox()
        self._wmax_edit.setRange(1, 1000)
        value = settings.value('segment/max_width',
                               50,
                               type=int)
        self._wmax_edit.setValue(value)
        layout.addRow(self._wmax_label, self._wmax_edit)
        self._hmin_label = qw.QLabel('Minimum length')
        self._hmin_edit = qw.QSpinBox()
        self._hmin_edit.setRange(1, 1000)
        value = settings.value('segment/min_height',
                               10,
                               type=int)
        self._hmin_edit.setValue(value)
        layout.addRow(self._hmin_label, self._hmin_edit)
        self._hmax_label = qw.QLabel('Maximum length')
        self._hmax_edit = qw.QSpinBox()
        self._hmax_edit.setRange(1, 1000)
        value = settings.value('segment/max_height',
                               100,
                               type=int)
        self._hmax_edit.setValue(value)
        layout.addRow(self._hmax_label, self._hmax_edit)
        self.setLayout(layout)
        self._hmax_edit.valueChanged.connect(self.sigHmax)
        self._hmin_edit.valueChanged.connect(self.sigHmin)
        self._wmax_edit.valueChanged.connect(self.sigWmax)
        self._wmin_edit.valueChanged.connect(self.sigWmin)
        self.sigQuit.connect(self.saveSettings)

    @qc.pyqtSlot()
    def saveSettings(self):
        """Save the worker parameters"""
        settings.setValue('segment/min_width', self._wmin_edit.value())
        settings.setValue('segment/max_width', self._wmax_edit.value())
        settings.setValue('segment/min_height', self._hmin_edit.value())
        settings.setValue('segment/max_height', self._hmax_edit.value())

    @qc.pyqtSlot(qg.QPolygonF)
    def setRoi(self, roi: qg.QPolygonF):
        self.roi = roi

    @qc.pyqtSlot()
    def resetRoi(self):
        self.roi = None

    @qc.pyqtSlot(np.ndarray, int)
    def process(self, bboxes: np.ndarray, pos: int) -> None:
        logging.debug(bboxes.shape)
        if len(bboxes) == 0:
            self.sigProcessed.emit(bboxes.copy(), pos)
            return
        wh = np.sort(bboxes[:, 2:], axis=1)
        valid = bboxes[(wh[:, 0] >= self._wmin_edit.value()) &
                       (wh[:, 0] <= self._wmax_edit.value()) &
                       (wh[:, 1] >= self._hmin_edit.value()) &
                       (wh[:, 1] <= self._hmax_edit.value())]
        if self.roi is None:
            self.sigProcessed.emit(valid.copy(), pos)
            return
        vidx = []
        for ii in range(valid.shape[0]):
            vertices = ut.rect2points(valid[ii, :])
            contained = [self.roi.containsPoint(qc.QPointF(*vtx),
                                                qc.Qt.OddEvenFill)
                         for vtx in vertices]
            if np.any(contained):
                vidx.append(ii)
        self.sigProcessed.emit(valid[vidx].copy(), pos)
