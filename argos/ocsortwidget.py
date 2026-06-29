# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
"""Qt widget wrapping OCSORTracker for use in the Argos GUI."""
import logging
import time
import numpy as np
from PyQt5 import QtCore as qc, QtWidgets as qw

from argos.ocsortracker import OCSORTracker, settings


class QOCSORTracker(qc.QObject):
    """Qt wrapper around :class:`OCSORTracker` for signal/slot communication."""

    sigTracked = qc.pyqtSignal(dict, int)

    def __init__(self, **kwargs):
        super().__init__()
        self.tracker = OCSORTracker(**kwargs)
        self._mutex = qc.QMutex()

    @qc.pyqtSlot()
    def reset(self) -> None:
        _ = qc.QMutexLocker(self._mutex)
        self.tracker.reset()

    @qc.pyqtSlot(float)
    def setIouThreshold(self, value: float) -> None:
        _ = qc.QMutexLocker(self._mutex)
        settings.setValue('ocsortracker/iou_threshold', value)
        self.tracker.iou_threshold = value

    @qc.pyqtSlot(int)
    def setMaxAge(self, value: int) -> None:
        _ = qc.QMutexLocker(self._mutex)
        settings.setValue('ocsortracker/max_age', value)
        self.tracker.max_age = value

    @qc.pyqtSlot(int)
    def setMinHits(self, value: int) -> None:
        _ = qc.QMutexLocker(self._mutex)
        settings.setValue('ocsortracker/min_hits', value)
        self.tracker.min_hits = value

    @qc.pyqtSlot(float)
    def setInertia(self, value: float) -> None:
        _ = qc.QMutexLocker(self._mutex)
        settings.setValue('ocsortracker/inertia', value)
        self.tracker.inertia = value

    @qc.pyqtSlot(int)
    def setDeltaT(self, value: int) -> None:
        _ = qc.QMutexLocker(self._mutex)
        settings.setValue('ocsortracker/delta_t', value)
        self.tracker.delta_t = value

    @qc.pyqtSlot(np.ndarray, int)
    def track(self, bboxes: np.ndarray, pos: int) -> None:
        _ts = time.perf_counter()
        _ = qc.QMutexLocker(self._mutex)
        result = {} if len(bboxes) == 0 else self.tracker.update(bboxes)
        logging.debug(f'OCSORTracker: frame {pos}, tracks: {result}')
        self.sigTracked.emit(result, pos)
        logging.debug(
            f'OCSORTracker.track: {time.perf_counter() - _ts:.4f}s'
        )


class OCSORTWidget(qw.QWidget):
    """Settings panel + threading wrapper for :class:`OCSORTracker`.

    Same signal/slot interface as
    :class:`~argos.sortrackerwidget.SORTWidget` and
    :class:`~argos.bytetrackerwidget.ByteTrackerWidget`.
    """

    sigTrack = qc.pyqtSignal(np.ndarray, int)
    sigTracked = qc.pyqtSignal(dict, int)
    sigQuit = qc.pyqtSignal()
    sigReset = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        iou_val = settings.value('ocsortracker/iou_threshold', 0.3, type=float)
        min_hits_val = settings.value('ocsortracker/min_hits', 3, type=int)
        max_age_val = settings.value('ocsortracker/max_age', 30, type=int)
        inertia_val = settings.value('ocsortracker/inertia', 0.2, type=float)
        delta_t_val = settings.value('ocsortracker/delta_t', 3, type=int)

        self._iou_label = qw.QLabel('Minimum IoU overlap')
        self._iou_label.setToolTip(
            'Minimum IoU between predicted and detected bbox to match them'
        )
        self._iou_spin = qw.QDoubleSpinBox()
        self._iou_spin.setRange(0.01, 1.0)
        self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setValue(iou_val)
        self._iou_spin.setToolTip(self._iou_label.toolTip())
        try:
            self._iou_spin.setStepType(qw.QDoubleSpinBox.AdaptiveDecimalStepType)
        except AttributeError:
            pass

        self._min_hits_label = qw.QLabel('Minimum hits')
        self._min_hits_label.setToolTip(
            'Consecutive detections needed before a track is confirmed'
        )
        self._min_hits_spin = qw.QSpinBox()
        self._min_hits_spin.setRange(1, 100)
        self._min_hits_spin.setValue(min_hits_val)
        self._min_hits_spin.setToolTip(self._min_hits_label.toolTip())

        self._max_age_label = qw.QLabel('Maximum age')
        self._max_age_label.setToolTip(
            'Frames a lost track is kept before deletion'
        )
        self._max_age_spin = qw.QSpinBox()
        self._max_age_spin.setRange(1, 500)
        self._max_age_spin.setValue(max_age_val)
        self._max_age_spin.setToolTip(self._max_age_label.toolTip())

        self._inertia_label = qw.QLabel('OCM inertia')
        self._inertia_label.setToolTip(
            'Weight of the velocity-direction cost (OCM). '
            '0 = pure IoU, higher values penalise direction reversals more.'
        )
        self._inertia_spin = qw.QDoubleSpinBox()
        self._inertia_spin.setRange(0.0, 1.0)
        self._inertia_spin.setSingleStep(0.05)
        self._inertia_spin.setValue(inertia_val)
        self._inertia_spin.setToolTip(self._inertia_label.toolTip())

        self._delta_t_label = qw.QLabel('Velocity window (Δt)')
        self._delta_t_label.setToolTip(
            'Number of frames back used to estimate motion direction. '
            'Larger values give smoother but more lagged velocity.'
        )
        self._delta_t_spin = qw.QSpinBox()
        self._delta_t_spin.setRange(1, 30)
        self._delta_t_spin.setValue(delta_t_val)
        self._delta_t_spin.setToolTip(self._delta_t_label.toolTip())

        self._disable_check = qw.QCheckBox('Disable tracking')
        self._disable_check.setToolTip(
            'Show detections without persistent track IDs (useful for debugging)'
        )

        layout = qw.QFormLayout()
        self.setLayout(layout)
        layout.addRow(self._iou_label, self._iou_spin)
        layout.addRow(self._min_hits_label, self._min_hits_spin)
        layout.addRow(self._max_age_label, self._max_age_spin)
        layout.addRow(self._inertia_label, self._inertia_spin)
        layout.addRow(self._delta_t_label, self._delta_t_spin)
        layout.addWidget(self._disable_check)

        self.qtracker = QOCSORTracker(
            iou_threshold=iou_val,
            min_hits=min_hits_val,
            max_age=max_age_val,
            inertia=inertia_val,
            delta_t=delta_t_val,
        )
        self.thread = qc.QThread()
        self.qtracker.moveToThread(self.thread)

        self._iou_spin.valueChanged.connect(self.qtracker.setIouThreshold)
        self._min_hits_spin.valueChanged.connect(self.qtracker.setMinHits)
        self._max_age_spin.valueChanged.connect(self.qtracker.setMaxAge)
        self._inertia_spin.valueChanged.connect(self.qtracker.setInertia)
        self._delta_t_spin.valueChanged.connect(self.qtracker.setDeltaT)
        self._disable_check.stateChanged.connect(self._disable)
        self.sigTrack.connect(self.qtracker.track)
        self.qtracker.sigTracked.connect(self.sigTracked)
        self.sigReset.connect(self.qtracker.reset)
        self.sigQuit.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @qc.pyqtSlot(int)
    def _disable(self, state: int) -> None:
        self.sigTrack.disconnect()
        if state:
            self.sigTrack.connect(self._send_dummy)
        else:
            self.sigTrack.connect(self.qtracker.track)

    @qc.pyqtSlot(np.ndarray, int)
    def _send_dummy(self, bboxes: np.ndarray, pos: int) -> None:
        result = {ii + 1: bboxes[ii] for ii in range(bboxes.shape[0])}
        self.sigTracked.emit(result, pos)

    @qc.pyqtSlot(np.ndarray, int)
    def track(self, bboxes: np.ndarray, pos: int) -> None:
        logging.debug(f'OCSORTWidget.track: frame {pos}')
        self.sigTrack.emit(bboxes, pos)
