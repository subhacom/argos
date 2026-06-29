# -*- coding: utf-8 -*-
# Author: Subhasis Ray
"""
====================================
Interface to YOLOv11 for segmentation
====================================

Replaces YOLACT with Ultralytics YOLOv11 instance segmentation.
The public signal/slot interface is identical to YolactWidget so the
rest of the pipeline (LimitsWidget, SORTWidget, VideoWidget) is unchanged.

Install the backend with::

    pip install ultralytics

References
----------
Ultralytics YOLOv11: https://docs.ultralytics.com
"""

import os
import logging
import time
import threading
import numpy as np

import torch
from PyQt5 import QtCore as qc, QtWidgets as qw

from argos.utility import init

settings = init()

_MODEL_VARIANTS = [
    ('Nano  (~3 MB,  fastest)',   'yolo11n-seg.pt'),
    ('Small (~10 MB, fast)',      'yolo11s-seg.pt'),
    ('Medium (~22 MB, balanced)', 'yolo11m-seg.pt'),
    ('Large (~25 MB, accurate)',  'yolo11l-seg.pt'),
    ('XLarge (~57 MB, best)',     'yolo11x-seg.pt'),
]


class Yolov11Exception(Exception):
    pass


class Yolov11Worker(qc.QObject):
    """Runs YOLOv11 inference in a QThread.

    Emits sigProcessed(bboxes, pos) where bboxes is an (N,4) int array in
    (x, y, w, h) format — identical to what YolactWorker emits.
    """

    sigProcessed = qc.pyqtSignal(np.ndarray, int)
    sigInitialized = qc.pyqtSignal()
    sigError = qc.pyqtSignal(Yolov11Exception)

    def __init__(self):
        super().__init__()
        self.mutex = qc.QMutex()
        self.top_k = 10
        self.score_threshold = 0.25
        self.overlap_thresh = 0.45  # NMS IOU threshold
        self.weights_file = ''
        self.net = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def setWaitCond(self, waitCond: threading.Event) -> None:
        _ = qc.QMutexLocker(self.mutex)
        self._waitCond = waitCond

    @qc.pyqtSlot(bool)
    def enableCuda(self, on: bool) -> None:
        settings.setValue('yolov11/cuda', on)
        self.device = 'cuda' if on and torch.cuda.is_available() else 'cpu'

    @qc.pyqtSlot(int)
    def setTopK(self, value: int) -> None:
        _ = qc.QMutexLocker(self.mutex)
        self.top_k = value

    @qc.pyqtSlot(float)
    def setScoreThresh(self, value: float) -> None:
        _ = qc.QMutexLocker(self.mutex)
        self.score_threshold = value

    @qc.pyqtSlot(float)
    def setOverlapThresh(self, value: float) -> None:
        _ = qc.QMutexLocker(self.mutex)
        self.overlap_thresh = value

    @qc.pyqtSlot(str)
    def setWeights(self, filename: str) -> None:
        if not filename:
            self.sigError.emit(Yolov11Exception('Empty weights filename'))
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            self.sigError.emit(Yolov11Exception(
                'ultralytics not installed. Run: pip install ultralytics'
            ))
            return
        tic = time.perf_counter_ns()
        try:
            self.net = YOLO(filename)
            # warm-up: prevents slow first-frame inference
            self.net.predict(
                np.zeros((64, 64, 3), dtype=np.uint8),
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            self.net = None
            self.sigError.emit(Yolov11Exception(str(exc)))
            return
        toc = time.perf_counter_ns()
        logging.debug('YOLOv11 model loaded in %.3f s', 1e-9 * (toc - tic))
        self.weights_file = filename
        self.sigInitialized.emit()

    @qc.pyqtSlot(np.ndarray, int)
    def process(self, image: np.ndarray, pos: int) -> None:
        """Detect objects; emit bboxes as (x, y, w, h) int array."""
        if self.net is None:
            self.sigError.emit(Yolov11Exception('Network not initialized'))
            return
        tic = time.perf_counter_ns()
        _ = qc.QMutexLocker(self.mutex)
        try:
            results = self.net.predict(
                image,
                conf=self.score_threshold,
                iou=self.overlap_thresh,
                max_det=self.top_k,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            self.sigError.emit(Yolov11Exception(str(exc)))
            return
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            self.sigProcessed.emit(np.empty((0, 4), dtype=np.int_), pos)
            return
        # Convert xyxy → x, y, w, h (same format as YolactWorker output)
        boxes = results[0].boxes.xyxy.cpu().numpy().copy()
        boxes[:, 2:] -= boxes[:, :2]
        boxes = np.rint(boxes).astype(np.int_)
        toc = time.perf_counter_ns()
        logging.debug('YOLOv11 frame %d processed in %.3f s', pos, 1e-9 * (toc - tic))
        self.sigProcessed.emit(boxes, pos)


class Yolov11Widget(qw.QWidget):
    """UI panel for YOLOv11 settings.

    Public interface is identical to YolactWidget:
      - sigProcessed(np.ndarray, int)   emitted after each frame
      - sigProcess(np.ndarray, int)     send a frame for processing
      - sigQuit()                       shut down the worker thread
      - process(image, pos)             slot — call to process a frame
      - initialized                     bool property
    """

    sigProcessed = qc.pyqtSignal(np.ndarray, int)
    sigProcess = qc.pyqtSignal(np.ndarray, int)
    sigTopK = qc.pyqtSignal(int)
    sigScoreThresh = qc.pyqtSignal(float)
    sigOverlapThresh = qc.pyqtSignal(float)
    sigWeightsFile = qc.pyqtSignal(str)
    sigQuit = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker = Yolov11Worker()
        self.initialized = False
        self.indicator = None

        # --- Standard model variant selector ---
        self.model_combo = qw.QComboBox()
        for label, _ in _MODEL_VARIANTS:
            self.model_combo.addItem(label)
        saved_idx = settings.value('yolov11/model_variant', 0, type=int)
        self.model_combo.setCurrentIndex(saved_idx)
        self.model_combo.setToolTip(
            'Standard pretrained model. Larger = more accurate but slower.\n'
            'Downloaded automatically from Ultralytics on first use.'
        )
        self.model_combo_label = qw.QLabel('Standard model')

        self.use_standard_action = qw.QAction('Use standard model', self)
        self.use_standard_action.setToolTip(
            'Download (if needed) and load the selected standard model'
        )
        self.use_standard_action.triggered.connect(self.loadStandardModel)

        # --- Custom weights loader ---
        self.load_weights_action = qw.QAction('Load custom weights', self)
        self.load_weights_action.setToolTip(
            'Load a .pt file trained on your own data with Ultralytics'
        )
        self.load_weights_action.triggered.connect(self.loadWeights)
        self.weights_label = qw.QLabel('(no model loaded)')
        self.weights_label.setToolTip('Active weights file')

        # --- CUDA toggle ---
        self.cuda_action = qw.QAction('Use CUDA', self)
        self.cuda_action.setCheckable(True)
        if torch.cuda.is_available():
            cuda_on = settings.value('yolov11/cuda', True, type=bool)
            self.cuda_action.setChecked(cuda_on)
            self.worker.device = 'cuda' if cuda_on else 'cpu'
            self.cuda_action.triggered.connect(self.worker.enableCuda)
            self.cuda_action.setToolTip('Use GPU acceleration via CUDA')
        else:
            self.cuda_action.setEnabled(False)
            self.worker.device = 'cpu'
            self.cuda_action.setToolTip('CUDA not available on this system')

        # --- Top K ---
        self.top_k_label = qw.QLabel('Max detections')
        self.top_k_edit = qw.QSpinBox()
        self.top_k_edit.setRange(1, 1000)
        saved_val = settings.value('yolov11/top_k', 10, type=int)
        self.top_k_edit.setValue(saved_val)
        self.worker.top_k = saved_val
        self.top_k_edit.setToolTip('Maximum number of objects to detect per frame')
        self.top_k_label.setToolTip(self.top_k_edit.toolTip())
        self.top_k_edit.valueChanged.connect(self.setTopK)

        # --- Confidence threshold ---
        self.score_thresh_label = qw.QLabel('Confidence threshold')
        self.score_thresh_edit = qw.QDoubleSpinBox()
        self.score_thresh_edit.setRange(0.01, 1.0)
        self.score_thresh_edit.setSingleStep(0.05)
        try:
            self.score_thresh_edit.setStepType(
                qw.QDoubleSpinBox.AdaptiveDecimalStepType
            )
        except AttributeError:
            pass
        saved_val = settings.value('yolov11/score_thresh', 0.25, type=float)
        self.score_thresh_edit.setValue(saved_val)
        self.worker.score_threshold = saved_val
        self.score_thresh_edit.setToolTip(
            'Minimum confidence to accept a detection (0–1). '
            'Higher = fewer but more certain detections.'
        )
        self.score_thresh_label.setToolTip(self.score_thresh_edit.toolTip())
        self.score_thresh_edit.valueChanged.connect(self.setScoreThresh)

        # --- NMS IOU threshold ---
        self.overlap_thresh_label = qw.QLabel('NMS overlap threshold')
        self.overlap_thresh_edit = qw.QDoubleSpinBox()
        self.overlap_thresh_edit.setRange(0.01, 1.0)
        self.overlap_thresh_edit.setSingleStep(0.05)
        try:
            self.overlap_thresh_edit.setStepType(
                qw.QDoubleSpinBox.AdaptiveDecimalStepType
            )
        except AttributeError:
            pass
        saved_val = settings.value('yolov11/overlap_thresh', 0.45, type=float)
        self.overlap_thresh_edit.setValue(saved_val)
        self.worker.overlap_thresh = saved_val
        self.overlap_thresh_edit.setToolTip(
            'Two detections with IOU above this are merged by NMS. '
            'Lower values remove more duplicate detections.'
        )
        self.overlap_thresh_label.setToolTip(self.overlap_thresh_edit.toolTip())
        self.overlap_thresh_edit.valueChanged.connect(self.setOverlapThresh)

        # --- Layout ---
        layout = qw.QFormLayout()
        self.setLayout(layout)

        layout.addRow(self.model_combo_label, self.model_combo)
        btn = qw.QToolButton()
        btn.setDefaultAction(self.use_standard_action)
        layout.addRow(btn)

        btn = qw.QToolButton()
        btn.setDefaultAction(self.load_weights_action)
        layout.addRow(btn, self.weights_label)

        btn = qw.QToolButton()
        btn.setDefaultAction(self.cuda_action)
        layout.addRow(btn)

        layout.addRow(self.top_k_label, self.top_k_edit)
        layout.addRow(self.score_thresh_label, self.score_thresh_edit)
        layout.addRow(self.overlap_thresh_label, self.overlap_thresh_edit)

        # --- Threading ---
        self.thread = qc.QThread()
        self.worker.moveToThread(self.thread)

        # --- Signal connections ---
        self.sigProcess.connect(self.worker.process)
        self.worker.sigError.connect(self.showError)
        self.worker.sigProcessed.connect(self.sigProcessed)
        self.worker.sigInitialized.connect(self.setInitialized)
        self.sigScoreThresh.connect(self.worker.setScoreThresh)
        self.sigOverlapThresh.connect(self.worker.setOverlapThresh)
        self.sigWeightsFile.connect(self.worker.setWeights)
        self.sigTopK.connect(self.worker.setTopK)
        self.sigQuit.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @qc.pyqtSlot(Yolov11Exception)
    def showError(self, err: Yolov11Exception) -> None:
        qw.QMessageBox.critical(self, 'YOLOv11 error', str(err))

    @qc.pyqtSlot(int)
    def setTopK(self, value: int) -> None:
        settings.setValue('yolov11/top_k', value)
        self.sigTopK.emit(value)

    @qc.pyqtSlot(float)
    def setScoreThresh(self, value: float) -> None:
        settings.setValue('yolov11/score_thresh', value)
        self.sigScoreThresh.emit(value)

    @qc.pyqtSlot(float)
    def setOverlapThresh(self, value: float) -> None:
        settings.setValue('yolov11/overlap_thresh', value)
        self.sigOverlapThresh.emit(value)

    @qc.pyqtSlot()
    def loadStandardModel(self) -> None:
        idx = self.model_combo.currentIndex()
        settings.setValue('yolov11/model_variant', idx)
        _, model_name = _MODEL_VARIANTS[idx]
        self.weights_label.setText(model_name)
        self._startLoading(model_name)

    @qc.pyqtSlot()
    def loadWeights(self) -> None:
        directory = settings.value('yolov11/weightsdir', '.')
        filename, ok = qw.QFileDialog.getOpenFileName(
            self,
            'Open YOLOv11 weights file',
            directory=directory,
            filter='Weights file (*.pt)',
        )
        if not filename or not ok:
            return
        settings.setValue('yolov11/weightsdir', os.path.dirname(filename))
        settings.setValue('yolov11/weightsfile', filename)
        self.weights_label.setText(os.path.basename(filename))
        self.weights_label.setToolTip(filename)
        self._startLoading(filename)

    def _startLoading(self, weights: str) -> None:
        self.initialized = False
        if self.indicator is None:
            self.indicator = qw.QProgressDialog(
                'Loading YOLOv11 model…', 'Cancel', 0, 0, self
            )
            self.indicator.setWindowModality(qc.Qt.WindowModal)
        try:
            self.worker.sigInitialized.disconnect()
        except TypeError:
            pass
        self.worker.sigInitialized.connect(self.indicator.reset)
        self.worker.sigInitialized.connect(self.setInitialized)
        self.indicator.show()
        self.sigWeightsFile.emit(weights)

    @qc.pyqtSlot()
    def setInitialized(self) -> None:
        self.initialized = True

    @qc.pyqtSlot(np.ndarray, int)
    def process(self, image: np.ndarray, pos: int) -> None:
        """Entry point from VideoWidget.sigSetFrame.

        Auto-loads the selected standard model on first call if no weights
        have been explicitly set.
        """
        if self.worker.net is None:
            try:
                self.loadStandardModel()
            except Exception as err:
                qw.QMessageBox.critical(self, 'Could not load model', str(err))
            return
        if self.initialized:
            self.sigProcess.emit(image, pos)
