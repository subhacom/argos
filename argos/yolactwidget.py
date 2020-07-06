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

from argos.utility import init


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

    def __init__(self):
        super(YolactWorker, self).__init__()
        self.mutex = qc.QMutex()
        self._image = None
        self._pos = 0
        self.top_k = 10
        self.cuda = torch.cuda.is_available()
        self.net = None
        self.score_threshold = 0.15
        self.config = yconfig.cfg
        self.weights_file = ''
        self.config_file = ''
        self.video_file = None

    def setWaitCond(self, waitCond: threading.Event) -> None:
        _ = qc.QMutexLocker(self.mutex)
        self._waitCond = waitCond

    @qc.pyqtSlot(bool)
    def enableCuda(self, on):
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
            raise YolactException('Network not initialized')
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
            # num_dets_to_consider = min(self.top_k, classes.shape[0])
            # for j in range(num_dets_to_consider):
            #     if scores[j] < self.score_threshold:
            #         num_dets_to_consider = j
            #         break
            # logging.debug('Bounding boxes: %r', boxes)
            # Convert from top-left bottom-right format to
            # top-left, width, height format
            boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]
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
    sigConfigFile = qc.pyqtSignal(str)
    sigWeightsFile = qc.pyqtSignal(str)
    sigQuit = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(YolactWidget, self).__init__(*args, **kwargs)
        self.worker = YolactWorker()
        self.load_config_action = qw.QAction('Load YOLACT configuration', self)
        self.load_config_action.setToolTip('Load YOLACT configuration. This '
                                           'should be a YAML (.yml) file '
                                           'containing key value pairs for '
                                           'various parameters for YOLACT')
        self.load_weights_action = qw.QAction('Load YOLACT weights', self)
        self.load_weights_action.setToolTip('Load the trained connection weights'
                                           ' for the YOLACT neural network.')
        self.load_config_action.triggered.connect(self.loadConfig)
        self.load_weights_action.triggered.connect(self.loadWeights)
        self.cuda_action = qw.QAction('Use CUDA')
        self.cuda_action.setCheckable(True)
        if torch.cuda.is_available():
            self.cuda_action.setChecked(self.worker.cuda)
            self.cuda_action.triggered.connect(self.worker.enableCuda)
            self.cuda_action.setToolTip('Try to use GPU if available')
        else:
            self.cuda_action.setEnabled(False)
            self.cuda_action.setToolTip('PyTorch on this system does not '
                                        'support CUDA')
        self.top_k_edit = qw.QLineEdit()
        saved_val = settings.value('yolact/topk', '10')
        self.top_k_edit.setText(saved_val)
        self.worker.top_k = int(saved_val)

        self.top_k_edit.editingFinished.connect(self.setTopK)
        self.top_k_edit.setToolTip('Include only this many objects'
                                   ' from all that are detected, ordered'
                                   ' by their classification score')
        self.top_k_label = qw.QLabel('Number of objects to include')
        self.top_k_label.setToolTip(self.top_k_edit.toolTip())
        self.score_thresh_edit = qw.QLineEdit()
        saved_val = settings.value('yolact/scorethreshold', '0.15')
        self.score_thresh_edit.setText(saved_val)
        self.worker.score_threshold = float(saved_val)

        self.score_thresh_edit.editingFinished.connect(self.setScoreThresh)
        self.score_thresh_edit.setToolTip('a number > 0 and < 1. Higher score'
                                          ' is more stringent criterion for'
                                          ' classifying objects')
        self.score_thresh_label = qw.QLabel('Detection score minimum')
        self.score_thresh_label.setToolTip(self.score_thresh_edit.toolTip())
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
        ######################################################
        # Threading
        self.thread = qc.QThread()
        self.worker.moveToThread(self.thread)
        ######################################################
        # Setup connections
        ######################################################
        self.sigProcess.connect(self.worker.process)
        self.worker.sigProcessed.connect(self.sigProcessed)
        self.sigScoreThresh.connect(self.worker.setScoreThresh)
        self.sigConfigFile.connect(self.worker.setConfig)
        self.sigWeightsFile.connect(self.worker.setWeights)
        self.sigTopK.connect(self.worker.setTopK)
        self.sigQuit.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @qc.pyqtSlot()
    def setTopK(self):
        self.sigTopK.emit(int(self.top_k_edit.text()))

    @qc.pyqtSlot()
    def setScoreThresh(self):
        self.sigScoreThresh.emit(float(self.score_thresh_edit.text()))

    @qc.pyqtSlot()
    def loadConfig(self):
        directory = settings.value('yolact/configdir', '.')
        filename, _ = qw.QFileDialog.getOpenFileName(
            self,
            'Open YOLACT configuration file',
            directory=directory, filter='YAML file (*.yml *.yaml)')
        if len(filename) == 0:
            return
        settings.setValue('yolact/configdir', os.path.dirname(filename))
        self.sigConfigFile.emit(filename)

    @qc.pyqtSlot()
    def loadWeights(self):
        directory = settings.value('yolact/configdir', '.')
        filename, _ = qw.QFileDialog.getOpenFileName(self, 'Open trained model',
                                                     directory=directory,
                                                     filter='Weights file (*.pth)')
        if len(filename) == 0:
            return
        settings.setValue('yolact/configdir', os.path.dirname(filename))
        self.sigWeightsFile.emit(filename)
        indicator = qw.QProgressDialog('Setting up neural net', 'Cancel', 0, 0, self)
        indicator.setWindowModality(qc.Qt.WindowModal)
        self.worker.sigInitialized.connect(indicator.reset)
        indicator.show()

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
        self.sigProcess.emit(image, pos)

    def __del__(self):
        settings = qc.QSettings()
        settings.setValue('yolact/scorethreshold',
                          self.score_thresh_edit.text())
        settings.setValue('yolact/topk', self.top_k_edit.text())
