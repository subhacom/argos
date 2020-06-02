# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-01 5:05 PM
"""Interface to YOLACT for segmentation"""


import sys
import time
import logging
from queue import Queue
import numpy as np
import multiprocessing
import threading
import concurrent.futures as cf
import yaml
import cv2
import torch
import torch.backends.cudnn as cudnn

from PyQt5 import (
    QtCore as qc,
    QtGui as qg,
    QtWidgets as qw
)

from yolact.yolact import Yolact
from yolact.data import config as yconfig
# This is actually yolact.utils
from yolact.utils.augmentations import FastBaseTransform, Resize
from yolact.layers import output_utils as oututils

from argos.utility import init


settings = init()


class YolactThread(qc.QThread):
    # emits list of classes, scores, and bboxes of detected objects
    # bboxes are in (top-left, w, h) format
    # The even is passed for synchronizing display of image in videowidget
    # with the bounding boxes
    sigProcessed = qc.pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, waitCond=None):
        super(YolactThread, self).__init__()
        self.mutex = qc.QMutex()
        self._waitCond = waitCond
        self._image = None
        self._pos = 0
        self.stop = False
        self.top_k = 10
        self.cuda = True
        self.net = None
        self.score_threshold = 0.15
        self.config = yconfig.cfg
        self.weights_file = ''
        self.config_file = ''
        self.video_file = None

    def setWaitCond(self, waitCond: threading.Event) -> None:
        self._waitCond = waitCond

    @qc.pyqtSlot(bool)
    def enableCuda(self, on):
        self.cuda = on

    @qc.pyqtSlot(int)
    def setTopK(self, value):
        self.top_k = value

    @qc.pyqtSlot(int)
    def setBatchSize(self, value):
        self.batch_size = int(value)

    @qc.pyqtSlot(float)
    def setScoreThresh(self, value):
        self.score_threshold = value

    @qc.pyqtSlot()
    def stopBatch(self):
        _ = qc.QMutexLocker(self.mutex)
        self.stop = True
        logging.debug('!!!!! Stopped thread')

    @qc.pyqtSlot(str)
    def setConfig(self, filename):
        if filename == '':
            return
        self.config_file = filename
        _ = qc.QMutexLocker(self.mutex)
        with open(filename, 'r') as cfg_file:
            config = yaml.safe_load(cfg_file)
            for key, value in config.items():
                logging.debug('%r \n%r %r', key, type(value), value)
                self.config.__setattr__(key, value)
            if 'mask_proto_debug' not in config:
                self.config.mask_proto_debug = False
        logging.debug(yaml.dump(self.config))

    @qc.pyqtSlot(str)
    def setWeights(self, filename):
        if filename == '':
            raise Exception('Empty filename for network weights')
        self.weights_file = filename
        tic = time.perf_counter_ns()
        _ = qc.QMutexLocker(self.mutex)
        with torch.no_grad():
            if self.cuda:
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')
            self.net = Yolact()
            self.net.load_weights(self.weights_file)
            self.net.eval()
            if self.cuda:
                self.net = self.net.cuda()
        toc = time.perf_counter_ns()
        logging.debug('Time to load weights %f s', 1e-9 * (toc - tic))

    @qc.pyqtSlot(np.ndarray)
    def setImage(self, image: np.ndarray, pos: int) -> None:
        self.mutex.lock()
        self._pos = pos
        self._image = image.copy()
        self.mutex.unlock()
        self.start()

    def process(self, image: np.ndarray):
        """:returns (classes, scores, boxes)

        where `boxes` is an array of bounding boxes of detected objects in
        (xleft, ytop, width, height) format.

        `classes` is the class ids of the corresponding objects.

        `scores` are the computed class scores corresponding to the detected objects.
        Roughly high score indicates strong belief that the object belongs to
        the identified class.
        """
        if self.net is None:
            raise Exception('Network not initialized')
        # Partly follows yolact eval.py
        tic = time.perf_counter_ns()

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
            return classes, scores, boxes

    def run(self):
        self.mutex.lock()
        classes, scores, bboxes = self.process(self._image)
        logging.debug(f'Processed frame {self._pos}')
        self.mutex.unlock()
        self.sigProcessed.emit(classes, scores, bboxes)
        if self._waitCond is not None:
            self._waitCond.wait()


class YolactWidget(qw.QWidget):
    # pass on the signal from YolactThread
    sigProcessed = qc.pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    # pass on the signal from YolactThread
    sigFinished = qc.pyqtSignal()
    # Pass UI entries to worker YolactThread
    sigTopK = qc.pyqtSignal(int)
    sigScoreThresh = qc.pyqtSignal(float)
    sigConfigFile = qc.pyqtSignal(str)
    sigWeightsFile = qc.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(YolactWidget, self).__init__(*args, **kwargs)
        self.worker = YolactThread()
        self.load_config_action = qw.QAction('Load configuration', self)
        self.load_weights_action = qw.QAction('Load YOLACT weights', self)
        self.load_config_action.triggered.connect(self.loadConfig)
        self.load_weights_action.triggered.connect(self.loadWeights)
        self.cuda_action = qw.QAction('Use CUDA')
        self.cuda_action.setCheckable(True)
        self.cuda_action.setChecked(self.worker.cuda)
        self.cuda_action.triggered.connect(self.worker.enableCuda)
        self.top_k_edit = qw.QLineEdit('10')
        self.top_k_edit.setText(str(self.worker.top_k))
        self.top_k_edit.editingFinished.connect(self.setTopK)
        self.top_k_label = qw.QLabel('Number of objects to include')
        self.score_thresh_edit = qw.QLineEdit('10')
        self.score_thresh_edit.setText(str(self.worker.score_threshold))
        self.score_thresh_edit.editingFinished.connect(self.setScoreThresh)
        self.score_thresh_label = qw.QLabel('Detection score minimum')
        ######################################################
        # Organize the actions as buttons in a form layout
        ######################################################
        layout = qw.QFormLayout()
        self.setLayout(layout)
        button = qw.QToolButton()
        button.setDefaultAction(self.load_config_action)
        layout.addRow(button)
        button = qw.QToolButton()
        button.setDefaultAction(self.load_weights_action)
        layout.addRow(button)
        button = qw.QToolButton()
        button.setDefaultAction(self.cuda_action)
        layout.addRow(button)
        layout.addRow(self.top_k_label, self.top_k_edit)
        layout.addRow(self.score_thresh_label, self.score_thresh_edit)
        ######################################################
        # Setup connections
        ######################################################
        self.worker.sigProcessed.connect(self.sigProcessed)
        self.worker.finished.connect(self.sigFinished)
        self.sigScoreThresh.connect(self.worker.setScoreThresh)
        self.sigConfigFile.connect(self.worker.setConfig)
        self.sigWeightsFile.connect(self.worker.setWeights)
        self.sigTopK.connect(self.worker.setTopK)


    def setWaitCond(self, cond):
        self.worker.setWaitCond(cond)

    @qc.pyqtSlot()
    def setTopK(self):
        self.sigTopK.emit(int(self.top_k_edit.text()))

    @qc.pyqtSlot()
    def setScoreThresh(self):
        self.sigScoreThresh.emit(float(self.score_thresh_edit.text()))

    @qc.pyqtSlot()
    def loadConfig(self):
        filename, _ = qw.QFileDialog.getOpenFileName(self, 'Open configuration')
        if len(filename) == 0:
            raise Exception('Empty filename for YOLACT configuration')
        self.sigConfigFile.emit(filename)

    @qc.pyqtSlot()
    def loadWeights(self):
        filename, _ = qw.QFileDialog.getOpenFileName(self, 'Open trained model')
        if len(filename) == 0:
            return
        self.sigWeightsFile.emit(filename)

    @qc.pyqtSlot(np.ndarray, int)
    def process(self, image: np.ndarray, pos: int) -> None:
        """If network has not been instantiated, ask the user to provide
        configuration and weights.

        `pos` - for debugging - should be frame no.
        """
        if self.worker.net is None:
            try:
                self.loadConfig()
                self.loadWeights()
            except Exception as err:
                qw.QMessageBox.critical('Could not open file', str(err))
                return
        self.worker.setImage(image, pos)
        self.worker.start()
        # this causes single-threading and reduces ~0.02 second on my laptop
        # self.worker.run()
