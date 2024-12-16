# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-11-28 11:45 PM
import logging
import time
import numpy as np
from collections import OrderedDict
from PyQt5 import QtCore as qc, QtWidgets as qw

import argos.constants
from argos.sortracker import SORTracker, settings

# Ordered dict to maintain sequence
distance_metric = OrderedDict(
    (('Intersection over Union', argos.constants.DistanceMetric.iou),
     ('Euclidean', argos.constants.DistanceMetric.euclidean))
)


class QSORTracker(qc.QObject):
    """Qt Wrapper class over SORTracker to allow signal/slot communication"""
    sigTracked = qc.pyqtSignal(dict, int)

    def __init__(self, *args, **kwargs):
        super(QSORTracker, self).__init__()
        self.tracker = SORTracker(*args, **kwargs)
        self._mutex = qc.QMutex()

    @qc.pyqtSlot()
    def reset(self):
        _ = qc.QMutexLocker(self._mutex)
        self.tracker.reset()

    @qc.pyqtSlot(argos.constants.DistanceMetric)
    def setDistMetric(self, metric):
        self.tracker.setDistMetric(metric)

    @qc.pyqtSlot(float)
    def setMinDist(self, dist: float) -> None:
        _ = qc.QMutexLocker(self._mutex)
        self.tracker.setMinDist(dist)
        metric = settings.value('sortracker/metric', 'iou', type=str)
        if metric == 'iou':
            settings.setValue('sortracker/iou_mindist', dist)
        else:
            settings.setValue('sortracker/euclidean_mindist', dist)

    @qc.pyqtSlot(int)
    def setMaxAge(self, max_age: int) -> None:
        """Set the maximum misses before discarding a track"""
        _ = qc.QMutexLocker(self._mutex)
        settings.setValue('sortracker/max_age', max_age)
        self.tracker.max_age = max_age

    @qc.pyqtSlot(int)
    def setMinHits(self, count: int) -> None:
        """Number of times a track should match prediction before it is
        confirmed"""
        _ = qc.QMutexLocker(self._mutex)
        settings.setValue('sortracker/min_hits', count)
        self.tracker.n_init = count

    @qc.pyqtSlot(dict, int)
    def track(self, bboxes: np.ndarray, pos: int) -> None:
        _ts = time.perf_counter()
        logging.debug(f'Received from {self.sender()} bboxes: {bboxes}')
        _ = qc.QMutexLocker(self._mutex)
        if len(bboxes) == 0:
            ret = {}
        else:
            ret = self.tracker.update(bboxes)
        logging.debug(f'SORTracker: frame {pos}, Rectangles: \n{ret}')
        self.sigTracked.emit(ret, pos)
        _dt = time.perf_counter() - _ts
        logging.debug(f'{__name__}.{self.__class__.__name__}.track: Runtime: {_dt}s')

class SORTWidget(qw.QWidget):
    sigTrack = qc.pyqtSignal(np.ndarray, int)
    sigTracked = qc.pyqtSignal(dict, int)
    sigQuit = qc.pyqtSignal()
    sigReset = qc.pyqtSignal()
    sigDistMetric = qc.pyqtSignal(argos.constants.DistanceMetric)

    def __init__(self, *args, **kwargs):
        super(SORTWidget, self).__init__(*args, **kwargs)
        self._max_age_label = qw.QLabel('Maximum age')
        self._max_age_label.setToolTip(
            'Maximum number of misses before a track is removed')
        self._max_age_spin = qw.QSpinBox()
        self._max_age_spin.setRange(1, 100)
        value = settings.value('sortracker/max_age', 10, type=int)
        self._max_age_spin.setValue(value)
        self._max_age_spin.setToolTip(self._max_age_label.toolTip())
        self._conf_age_label = qw.QLabel('Minimum hits')
        self._conf_age_label.setToolTip(
            'Minimum number of hits before a track is confirmed')
        self._conf_age_spin = qw.QSpinBox()
        self._conf_age_spin.setRange(1, 100)
        value = settings.value('sortracker/min_hits', 3, type=int)
        self._conf_age_spin.setValue(value)
        self._conf_age_spin.setToolTip(self._conf_age_label.toolTip())

        self._dist_metric_label = qw.QLabel('Distance metric')
        self._dist_metric_combo = qw.QComboBox()
        self._dist_metric_combo.addItems(list(distance_metric.keys()))
        metric = settings.value('sortracker/metric', 'iou', type=str)
        self._min_dist_spin = qw.QDoubleSpinBox()
        try:
            self._min_dist_spin.setStepType(
                qw.QDoubleSpinBox.AdaptiveDecimalStepType)
        except AttributeError:
            pass     # older versions of Qt don't support this
        if metric == 'iou':
            self._min_dist_label = qw.QLabel('Minimum overlap')
            self._min_dist_spin.setRange(0.1, 1.0)
            self._min_dist_spin.setSingleStep(0.05)
            value = settings.value('sortracker/iou_mindist', 0.3, type=float)
            self._min_dist_spin.setValue(value)
            self._min_dist_spin.setToolTip(
                'Minimum overlap between bounding boxes '
                'to consider them same object.')
        else:
            self._min_dist_label = qw.QLabel('Maximum distance')
            self._min_dist_spin.setRange(1, 1000)
            self._min_dist_spin.setSingleStep(5)
            self._min_dist_spin.setValue(
                settings.value('sortracker/euclidean_mindist', 5, type=float))
            self._min_dist_spin.setToolTip(
                'Maximum distance between bounding boxes '
                'to consider them same object.')

        self._disable_check = qw.QCheckBox('Disable tracking')
        self._disable_check.setToolTip('Just show the identified objects. Can '
                                       'be useful for troubleshooting.')
        layout = qw.QFormLayout()
        self.setLayout(layout)
        layout.addRow(self._dist_metric_label, self._dist_metric_combo)
        layout.addRow(self._min_dist_label, self._min_dist_spin)
        layout.addRow(self._conf_age_label, self._conf_age_spin)
        layout.addRow(self._max_age_label, self._max_age_spin)
        layout.addWidget(self._disable_check)
        self.qtracker = QSORTracker(metric=argos.constants.DistanceMetric.iou,
                                    min_dist=self._min_dist_spin.value(),
                                    max_age=self._max_age_spin.value(),
                                    n_init=self._conf_age_spin.value())
        self.thread = qc.QThread()
        self.qtracker.moveToThread(self.thread)
        self._max_age_spin.valueChanged.connect(self.qtracker.setMaxAge)
        self._dist_metric_combo.currentTextChanged.connect(self.setDistMetric)
        self._min_dist_spin.valueChanged.connect(self.qtracker.setMinDist)
        self._conf_age_spin.valueChanged.connect(self.qtracker.setMinHits)
        self._disable_check.stateChanged.connect(self.disable)
        self.sigDistMetric.connect(self.qtracker.setDistMetric)
        self.sigTrack.connect(self.qtracker.track)
        self.qtracker.sigTracked.connect(self.sigTracked)
        self.sigReset.connect(self.qtracker.reset)
        self.sigQuit.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @qc.pyqtSlot(str)
    def setDistMetric(self, text):
        metric = distance_metric[text]
        if metric == argos.constants.DistanceMetric.iou:
            # print('SORT Metric set to IoU')
            settings.setValue('sortracker/metric', 'iou')
            self._min_dist_label.setText('Minimum overlap')
            self._min_dist_spin.setRange(0.01, 1)
            self._min_dist_spin.setValue(
                settings.value('sortracker/iou_mindist', 0.3, type=float))
            self._min_dist_spin.setToolTip(
                'Minimum overlap between bounding boxes '
                'to consider them same object.')
        else:
            settings.setValue('sortracker/metric', 'euclidean')
            self._min_dist_label.setText('Maximum distance')
            self._min_dist_spin.setRange(1, 1000)
            self._min_dist_spin.setValue(
                settings.value('sortracker/euclidean_mindist', 5, type=float))
            self._min_dist_spin.setToolTip(
                'Maximum distance between bounding boxes '
                'to consider them same object.')
        self.sigDistMetric.emit(metric)

    @qc.pyqtSlot(int)
    def disable(self, state):
        self.sigTrack.disconnect()
        if state:
            self.sigTrack.connect(self.sendDummySigTracked)
        else:
            self.sigTrack.connect(self.qtracker.track)

    @qc.pyqtSlot(np.ndarray, int)
    def sendDummySigTracked(self, bboxes: np.ndarray, pos: int) -> None:
        ret = {ii + 1: bboxes[ii] for ii in range(bboxes.shape[0])}
        self.sigTracked.emit(ret, pos)

    @qc.pyqtSlot(np.ndarray, int)
    def track(self, bboxes: np.ndarray, pos: int) -> None:
        """Just to intercept signal source for debugging"""
        logging.debug(
            f'Received frame {pos} from {self.sender()} bboxes: {bboxes}')
        self.sigTrack.emit(bboxes, pos)


def test():
    import sys
    app = qw.QApplication(sys.argv)
    win = SORTWidget()
    win.setMinimumSize(800, 600)
    # win.setWindowTitle('Argos - track animals in video')
    # win.showMaximized()
    # app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    test()
