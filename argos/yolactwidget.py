# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-01 5:05 PM
"""Interface to YOLACT for segmentation"""

import os
import time
import logging
import numpy as np
import threading
import yaml
import torch
import torch.backends.cudnn as cudnn

from PyQt5 import (
    QtCore as qc,
    QtWidgets as qw
)

from yolact import Yolact
from yolact.data import config as yconfig
# This is actually yolact.utils
from yolact.utils.augmentations import FastBaseTransform
from yolact.layers import output_utils as oututils

from argos.utility import init, pairwise_distance
from argos.constants import OutlineStyle, DistanceMetric

settings = init()


class YolactException(BaseException):
    def __init__(self, *args, **kwargs):
        super(YolactException, self).__init__(*args, **kwargs)


class YolactWorker(qc.QObject):
    # emits list of classes, scores, and bboxes of detected objects
    # bboxes are in (top-left, w, h) format
    # The even is passed for synchronizing display of image in videowidget
    # with the bounding boxes
    sigProcessed = qc.pyqtSignal(np.ndarray, int)
    sigInitialized = qc.pyqtSignal()
    sigError = qc.pyqtSignal(YolactException)

    def __init__(self):
        super(YolactWorker, self).__init__()
        self.mutex = qc.QMutex()
        self._image = None
        self._pos = 0
        self.top_k = 10
        self.cuda = torch.cuda.is_available()
        self.net = None
        self.score_threshold = 0.15
        self.overlap_thresh = 1.0
        self.config = yconfig.cfg
        self.weights_file = ''
        self.config_file = ''
        self.video_file = None

    def setWaitCond(self, waitCond: threading.Event) -> None:
        _ = qc.QMutexLocker(self.mutex)
        self._waitCond = waitCond

    @qc.pyqtSlot(bool)
    def enableCuda(self, on):
        settings.setValue('yolact/cuda', on)
        self.cuda = on

    @qc.pyqtSlot(int)
    def setTopK(self, value):
        _ = qc.QMutexLocker(self.mutex)
        self.top_k = value

    @qc.pyqtSlot(int)
    def setBatchSize(self, value):
        _ = qc.QMutexLocker(self.mutex)
        self.batch_size = int(value)

    @qc.pyqtSlot(float)
    def setScoreThresh(self, value):
        _ = qc.QMutexLocker(self.mutex)
        self.score_threshold = value

    @qc.pyqtSlot(float)
    def setOverlapThresh(self, value):
        """Merge objects if their bboxes overlap more than this."""
        _ = qc.QMutexLocker(self.mutex)
        self.overlap_thresh = value

    @qc.pyqtSlot(str)
    def setConfig(self, filename):
        if filename == '':
            return
        self.config_file = filename
        with open(filename, 'r') as cfg_file:
            config = yaml.safe_load(cfg_file)
            for key, value in config.items():
                logging.debug('%r \n%r %r', key, type(value), value)
                self.config.__setattr__(key, value)
            if 'mask_proto_debug' not in config:
                self.config.mask_proto_debug = False
        logging.debug(yaml.dump(self.config))

    @qc.pyqtSlot(str)
    def setWeights(self, filename: str) -> None:
        if filename == '':
            raise YolactException('Empty filename for network weights')
        self.weights_file = filename
        tic = time.perf_counter_ns()
        with torch.no_grad():
            if self.cuda:
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')
            self.net = Yolact()
            self.net.load_weights(self.weights_file, self.cuda)
            self.net.eval()
            if self.cuda:
                self.net = self.net.cuda()
        toc = time.perf_counter_ns()
        logging.debug('Time to load weights %f s', 1e-9 * (toc - tic))
        self.sigInitialized.emit()

    @qc.pyqtSlot(np.ndarray, int)
    def process(self, image: np.ndarray, pos: int):
        """:returns (classes, scores, boxes)

        where `boxes` is an array of bounding boxes of detected objects in
        (xleft, ytop, width, height) format.

        `classes` is the class ids of the corresponding objects.

        `scores` are the computed class scores corresponding to the detected objects.
        Roughly high score indicates strong belief that the object belongs to
        the identified class.
        """
        logging.debug(f'Received frame {pos}')
        if self.net is None:
            self.sigError.emit(YolactException('Network not initialized'))
            return
        # Partly follows yolact eval.py
        tic = time.perf_counter_ns()
        _ = qc.QMutexLocker(self.mutex)
        with torch.no_grad():
            if self.cuda:
                image = torch.from_numpy(image).cuda().float()
            else:
                image = torch.from_numpy(image).float()
            batch = FastBaseTransform()(image.unsqueeze(0))
            preds = self.net(batch)
            image_gpu = image / 255.0
            h, w, _ = image.shape
            save = self.config.rescore_bbox
            self.config.rescore_bbox = True
            classes, scores, boxes, masks = oututils.postprocess(
                preds, w, h,
                visualize_lincomb=False,
                crop_masks=True,
                score_threshold=self.score_threshold)
            idx = scores.argsort(0, descending=True)[:self.top_k]
            # if self.config.eval_mask_branch:
            #     masks = masks[idx]
            classes, scores, boxes = [x[idx].cpu().numpy()
                                      for x in (classes, scores, boxes)]
            # This is probably not required, `postprocess` uses
            # `score_thresh` already
            num_dets_to_consider = min(self.top_k, classes.shape[0])
            for j in range(num_dets_to_consider):
                if scores[j] < self.score_threshold:
                    num_dets_to_consider = j
                    break
            # logging.debug('Bounding boxes: %r', boxes)
            # Convert from top-left bottom-right format to
            # top-left, width, height format
            if len(boxes) == 0:
                self.sigProcessed.emit(boxes, pos)
                return
            boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]
            if self.overlap_thresh < 1:
                dist_matrix = pairwise_distance(new_bboxes=boxes, bboxes=boxes,
                                                boxtype=OutlineStyle.bbox,
                                                metric=DistanceMetric.ios)
                bad_idx = [jj for ii in range(dist_matrix.shape[0] - 1) \
                             for jj in range(ii+1, dist_matrix.shape[1]) \
                              if dist_matrix[ii, jj] < 1 - self.overlap_thresh]
                good_idx = list(set(range(boxes.shape[0])) - set(bad_idx))
                boxes = boxes[good_idx].copy()

            toc = time.perf_counter_ns()
            logging.debug('Time to process single _image: %f s',
                          1e-9 * (toc - tic))
            self.sigProcessed.emit(boxes, pos)
            logging.debug(f'Emitted bboxes for frame {pos}: {boxes}')


class YolactWidget(qw.QWidget):
    # pass on the signal from YolactWorker
    sigProcessed = qc.pyqtSignal(np.ndarray, int)
    # pass on the image to YolactWorker for processing
    sigProcess = qc.pyqtSignal(np.ndarray, int)

    # Pass UI entries to worker YolactWorker
    sigTopK = qc.pyqtSignal(int)
    sigScoreThresh = qc.pyqtSignal(float)
    sigOverlapThresh = qc.pyqtSignal(float)
    sigConfigFile = qc.pyqtSignal(str)
    sigWeightsFile = qc.pyqtSignal(str)
    sigQuit = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(YolactWidget, self).__init__(*args, **kwargs)
        self.worker = YolactWorker()
        self.initialized = False
        self.indicator = None
        self.load_config_action = qw.QAction('Load YOLACT configuration', self)
        self.load_config_action.setToolTip('Load YOLACT configuration. This '
                                           'should be a YAML (.yml) file '
                                           'containing key value pairs for '
                                           'various parameters for YOLACT')
        self.load_weights_action = qw.QAction('Load YOLACT weights', self)
        self.load_weights_action.setToolTip(
            'Load the trained connection weights'
            ' for the YOLACT neural network.')
        self.load_config_action.triggered.connect(self.loadConfig)
        self.load_weights_action.triggered.connect(self.loadWeights)
        self.cuda_action = qw.QAction('Use CUDA')
        self.cuda_action.setCheckable(True)
        if torch.cuda.is_available():
            self.cuda_action.setChecked(self.worker.cuda)
            settings.setValue('yolact/cuda', self.worker.cuda)
            self.cuda_action.triggered.connect(self.worker.enableCuda)
            self.cuda_action.setToolTip('Try to use GPU if available')
        else:
            self.cuda_action.setEnabled(False)
            settings.setValue('yolact/cuda', False)
            self.cuda_action.setToolTip('PyTorch on this system does not '
                                        'support CUDA')
        self.top_k_edit = qw.QSpinBox()
        self.top_k_edit.setRange(1, 1000)
        saved_val = settings.value('yolact/top_k', 10, type=int)
        self.top_k_edit.setValue(int(saved_val))
        self.worker.top_k = int(saved_val)

        self.top_k_edit.valueChanged.connect(self.setTopK)
        self.top_k_edit.setToolTip('Include only this many objects'
                                   ' from all that are detected, ordered'
                                   ' by their classification score')
        self.top_k_label = qw.QLabel('Number of objects to include')
        self.top_k_label.setToolTip(self.top_k_edit.toolTip())
        self.score_thresh_edit = qw.QDoubleSpinBox()
        saved_val = settings.value('yolact/score_thresh', 0.15, type=float)
        self.score_thresh_edit.setRange(0.01, 1.0)
        try:
            self.score_thresh_edit.setStepType(
                qw.QDoubleSpinBox.AdaptiveDecimalStepType)
        except AttributeError:
            pass     # older versions of Qt don't support this
        self.score_thresh_edit.setSingleStep(0.05)
        self.score_thresh_edit.setValue(float(saved_val))
        self.worker.score_threshold = float(saved_val)

        self.score_thresh_edit.valueChanged.connect(self.setScoreThresh)
        self.score_thresh_edit.setToolTip('a number > 0 and < 1. Higher score'
                                          ' is more stringent criterion for'
                                          ' classifying objects')
        self.score_thresh_label = qw.QLabel('Detection score minimum')
        self.score_thresh_label.setToolTip(self.score_thresh_edit.toolTip())

        self.overlap_thresh_label = qw.QLabel('Merge overlaps more than')
        self.overlap_thresh_edit = qw.QDoubleSpinBox()
        self.overlap_thresh_edit.setRange(0.01, 1.1)
        self.overlap_thresh_edit.setToolTip('a number > 0 and < 1. If the '
                                            'ratio of overlap between two objects '
                                            'and the bounding rectangle of the '
                                            'smaller object more than this, merge '
                                            'them into a single object ')
        # self.overlap_thresh_edit.setSingleStep(0.01)
        try:
            self.overlap_thresh_edit.setStepType(
                qw.QDoubleSpinBox.AdaptiveDecimalStepType)
        except AttributeError:
            pass  # older Qt versions
        saved_val = settings.value('yolact/overlap_thresh', 1.0, type=float)
        self.overlap_thresh_edit.setValue(saved_val)
        self.worker.setOverlapThresh(saved_val)
        self.overlap_thresh_edit.valueChanged.connect(self.setOverlapThresh)

        self.ignore = False
        ######################################################
        # Organize the actions as buttons in a form layout
        ######################################################
        layout = qw.QFormLayout()
        self.setLayout(layout)
        button = qw.QToolButton()
        button.setDefaultAction(self.load_config_action)
        button.setToolTip(self.load_config_action.toolTip())
        layout.addRow(button)
        button = qw.QToolButton()
        button.setDefaultAction(self.load_weights_action)
        button.setToolTip(self.load_weights_action.toolTip())
        layout.addRow(button)
        button = qw.QToolButton()
        button.setDefaultAction(self.cuda_action)
        button.setToolTip(self.cuda_action.toolTip())
        layout.addRow(button)
        layout.addRow(self.top_k_label, self.top_k_edit)
        layout.addRow(self.score_thresh_label, self.score_thresh_edit)
        layout.addRow(self.overlap_thresh_label, self.overlap_thresh_edit)
        ######################################################
        # Threading
        self.thread = qc.QThread()
        self.worker.moveToThread(self.thread)
        ######################################################
        # Setup connections
        ######################################################
        self.sigProcess.connect(self.worker.process)
        self.worker.sigError.connect(self.showYolactError)
        self.worker.sigProcessed.connect(self.sigProcessed)
        self.worker.sigInitialized.connect(self.setInitialized)
        self.sigScoreThresh.connect(self.worker.setScoreThresh)
        self.sigOverlapThresh.connect(self.worker.setOverlapThresh)
        self.sigConfigFile.connect(self.worker.setConfig)
        self.sigWeightsFile.connect(self.worker.setWeights)
        self.sigTopK.connect(self.worker.setTopK)
        self.sigQuit.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @qc.pyqtSlot(YolactException)
    def showYolactError(self, err: YolactException) -> None:
        qw.QMessageBox.critical(self, 'Yolact error', str(err))

    @qc.pyqtSlot(int)
    def setTopK(self, value):
        settings.setValue('yolact/top_k', value)
        self.sigTopK.emit(value)

    @qc.pyqtSlot(float)
    def setScoreThresh(self, value):
        settings.setValue('yolact/score_thresh', value)
        self.sigScoreThresh.emit(value)

    @qc.pyqtSlot(float)
    def setOverlapThresh(self, value):
        settings.setValue('yolact/overlap_thresh', value)
        self.sigOverlapThresh.emit(value)

    @qc.pyqtSlot()
    def loadConfig(self):
        directory = settings.value('yolact/configdir', '.')
        filename, ok = qw.QFileDialog.getOpenFileName(
            self,
            'Open YOLACT configuration file',
            directory=directory, filter='YAML file (*.yml *.yaml)')
        if len(filename) == 0 or not ok:
            return
        settings.setValue('yolact/configdir', os.path.dirname(filename))
        settings.setValue('yolact/configfile', filename)

        self.sigConfigFile.emit(filename)

    @qc.pyqtSlot()
    def loadWeights(self):
        directory = settings.value('yolact/configdir', '.')
        filename, ok = qw.QFileDialog.getOpenFileName(self,
                                                      'Open trained model',
                                                      directory=directory,
                                                      filter='Weights file (*.pth)')
        if len(filename) == 0 or not ok:
            return
        settings.setValue('yolact/configdir', os.path.dirname(filename))
        settings.setValue('yolact/weightsfile', filename)
        self.initialized = False
        self.sigWeightsFile.emit(filename)
        if self.indicator is None:
            self.indicator = qw.QProgressDialog('Setting up neural net',
                                                'Cancel', 0, 0, self)
            self.indicator.setWindowModality(qc.Qt.WindowModal)
        try:
            self.worker.sigInitialized.disconnect()
        except TypeError:
            pass
        self.worker.sigInitialized.connect(self.indicator.reset)
        self.worker.sigInitialized.connect(self.setInitialized)
        self.indicator.show()

    @qc.pyqtSlot()
    def setInitialized(self):
        self.initialized = True

    @qc.pyqtSlot(np.ndarray, int)
    def process(self, image: np.ndarray, pos: int) -> None:
        """If network has not been instantiated, ask the user to provide
        configuration and weights.

        `pos` - for debugging - should be frame no.
        """
        logging.debug(f'{self.__class__.__name__}: Receivec frame {pos}')
        if self.worker.net is None:
            try:
                self.loadConfig()
                self.loadWeights()
            except YolactException as err:
                qw.QMessageBox.critical(self, 'Could not open file', str(err))
                return
        if self.initialized:
            self.sigProcess.emit(image, pos)
