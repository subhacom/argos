# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-29 4:52 PM
"""Widget to generate training data for YOLACT"""
import sys
import time
import logging
import os
from collections import OrderedDict
import random
import pickle
from typing import Dict
from datetime import datetime
import numpy as np
import cv2
import json
import yaml
from PyQt5 import (
    QtCore as qc,
    QtWidgets as qw,
    QtGui as qg)

import argos.constants
from argos.constants import OutlineStyle
from argos import utility as ut
from argos import frameview
from argos.frameview import FrameView
from argos.segwidget import SegWidget
from argos.limitswidget import LimitsWidget
from yolact import config as yconfig

settings = ut.init()


class SegDisplay(FrameView):
    sigItemSelectionChanged = qc.pyqtSignal(list)
    sigPolygons = qc.pyqtSignal(dict)

    def __init__(self, *args, **kwargs):
        super(SegDisplay, self).__init__(*args, **kwargs)
        self.segList = qw.QListWidget()
        self.segList.setSizeAdjustPolicy(qw.QListWidget.AdjustToContents)
        self.segList.setSelectionMode(self.segList.ExtendedSelection)
        self.segList.itemSelectionChanged.connect(self.sendSelection)
        self.sigItemSelectionChanged.connect(self.scene().setSelected)
        self.keepSelectedAction = qw.QAction('Keep selected objects (K)')
        self.removeSelectedAction = qw.QAction('Remove selected objects (X)')
        self.keepSelectedAction.triggered.connect(self.scene().keepSelected)
        self.removeSelectedAction.triggered.connect(self.scene().removeSelected)
        self.scene().sigPolygons.connect(self.sigPolygons)
        self.scene().sigPolygons.connect(self.updateSegList)

    @qc.pyqtSlot()
    def sendSelection(self):
        selection = [int(item.text()) for item in
                     self.segList.selectedItems()]
        self.sigItemSelectionChanged.emit(selection)

    @qc.pyqtSlot(dict)
    def updateSegList(self, segdict: Dict[int, np.ndarray]) -> None:
        self.segList.clear()
        self.segList.addItems([str(key) for key in segdict.keys()])
        self.segList.updateGeometry()

    def setRoiMode(self):
        self.scene().setRoiPolygonMode()

    @qc.pyqtSlot(np.ndarray, int)
    def setBboxes(self, bboxes: Dict[int, np.ndarray], pos: int):
        """Method for converting x,y,w,h bbox into series of verices compatible
        with polygon settings"""
        polygons = {ii: ut.rect2points(bboxes[ii, :])
                    for ii in range(bboxes.shape[0])}
        self.setPolygons(polygons, pos)


class TrainingWidget(qw.QMainWindow):
    sigQuit = qc.pyqtSignal()
    # Send an image and its index in file list for segmentation
    sigSegment = qc.pyqtSignal(np.ndarray, int)
    # Send the image
    sigImage = qc.pyqtSignal(np.ndarray, int)
    # send refined segmentation data
    sigSegmented = qc.pyqtSignal(dict, int)
    # set geometry mode of drawing widget
    sigSetDisplayGeom = qc.pyqtSignal(argos.constants.DrawingGeom)

    def __init__(self, *args, **kwargs):
        super(TrainingWidget, self).__init__(*args, **kwargs)
        self._waiting = False
        self.boundaryType = 'contour'
        self.displayCoco = True
        self.numCrops = 1  # number of random crops to generate if input image is bigger than training image size
        self.saved = True
        self.validation_frac = 0.3
        self.description = ''
        self.license_name = ''
        self.license_url = ''
        self.contributor = ''
        self.category_name = 'object'
        self.url = ''
        self.inputImageSize = 550
        self.image_dir = settings.value('training/imagedir', '.')
        self.image_files = []
        self.image_index = -1
        self.training_dir = 'training'
        self.validation_dir = 'validation'
        self.out_dir = settings.value('training/outdir', '.')
        self.baseconfig_name = ''
        for name in dir(yconfig):
            if name.startswith('yolact') and name.endswith('config'):
                self.baseconfig_name = name
                break
        self.baseconfig = getattr(yconfig, self.baseconfig_name)
        self.weights_file = ''
        self.config_file = ''
        self.category_name = 'object'
        self.seg_dict = {}  # dict containing segmentation info for each file
        self.seg_widget = SegWidget()
        self.seg_widget.outline_combo.setCurrentText('contour')
        self.seg_widget.setOutlineStyle('contour')
        self.lim_widget = LimitsWidget()
        self.seg_dock = qw.QDockWidget('Segmentation settings')
        self.seg_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                      qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self.seg_dock)
        layout = qw.QVBoxLayout()
        layout.addWidget(self.seg_widget)
        layout.addWidget(self.lim_widget)
        widget = qw.QWidget()
        widget.setLayout(layout)
        scroll = qw.QScrollArea()
        scroll.setWidget(widget)
        self.seg_dock.setWidget(scroll)
        self.display_widget = SegDisplay()
        self.display_widget.setRoiMode()
        self.display_widget.frame_scene.linewidth = 1
        self.setCentralWidget(self.display_widget)
        self._makeActions()
        self._makeFileDock()
        self._makeSegDock()
        self._makeMenuBar()
        self.sigImage.connect(self.display_widget.setFrame)
        self.sigSegment.connect(self.seg_widget.sigProcess)
        self.seg_widget.sigSegPolygons.connect(
            self.display_widget.sigSetPolygons)
        self.display_widget.sigPolygons.connect(self.setSegmented)
        self.seg_widget.sigProcessed.connect(self.display_widget.setBboxes)
        self.lim_widget.sigWmin.connect(self.seg_widget.setWmin)
        self.lim_widget.sigWmax.connect(self.seg_widget.setWmax)
        self.lim_widget.sigHmin.connect(self.seg_widget.setHmin)
        self.lim_widget.sigHmax.connect(self.seg_widget.setHmax)
        # Note the difference between `sigSegment` and `sigSegmented`
        # - this TrainingWidget's `sigSegment` sends the image to the
        #   segmentation widget
        # - segmentation widget's `sigSegPolygons` sends the segmented polygons
        #   to display widget
        # - display widget passes the segmented polygon dict to this
        #   TrainingWidget via `sigPolygons` into `setSegmented` slot
        # - if seg widget is passing bboxes, then it sends them via
        #   `sigProcessed` into display widget's `setBboxes` slot
        #   - display widget's setBboxes slot converts the rects into polygon
        #       vtx and passes them via `sigPolygons`
        # - when the frame has been already segmented and is available in
        #   `seg_dict`, `sigSegmented` sends segmented polygons to
        #   displaywidget's `setPolygons` slot directly
        self.sigSegmented.connect(self.display_widget.setPolygons)
        self.sigQuit.connect(self.seg_widget.sigQuit)
        self.sigQuit.connect(self.lim_widget.sigQuit)
        self._makeShortcuts()
        self.openImageDir()
        self.statusBar().showMessage('Press `Next image` to start segmenting')

    def _makeFileDock(self):
        self.file_dock = qw.QDockWidget('Files/Dirs')

        dirlayout = qw.QFormLayout()
        self.out_dir_label = qw.QLabel('Output directory for training data')
        self.out_dir_name = qw.QLabel(self.out_dir)
        dirlayout.addRow(self.out_dir_label, self.out_dir_name)
        self.image_dir_label = qw.QLabel('Input image directory')
        self.image_dir_name = qw.QLabel(self.image_dir)
        dirlayout.addRow(self.image_dir_label, self.image_dir_name)
        self.dir_widget = qw.QWidget()
        self.dir_widget.setLayout(dirlayout)

        self.file_view = qw.QListView()
        self.file_view.setSizeAdjustPolicy(qw.QListWidget.AdjustToContents)
        self.file_model = qw.QFileSystemModel()
        self.file_model.setFilter(qc.QDir.NoDotAndDotDot | qc.QDir.Files)
        self.file_view.setModel(self.file_model)
        self.file_view.setRootIndex(self.file_model.setRootPath(self.image_dir))
        self.file_view.selectionModel().selectionChanged.connect(self.handleFileSelectionChanged)

        self.fwidget = qw.QWidget()
        layout = qw.QVBoxLayout()
        layout.addWidget(self.dir_widget)
        layout.addWidget(self.file_view)
        self.fwidget.setLayout(layout)
        self.file_dock.setWidget(self.fwidget)
        self.file_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                       qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self.file_dock)

    def _makeSegDock(self):
        self.next_button = qw.QToolButton()
        self.next_button.setSizePolicy(qw.QSizePolicy.Minimum,
                                       qw.QSizePolicy.MinimumExpanding)
        self.next_button.setDefaultAction(self.nextFrameAction)

        self.prev_button = qw.QToolButton()
        self.prev_button.setSizePolicy(qw.QSizePolicy.Minimum,
                                       qw.QSizePolicy.MinimumExpanding)
        self.prev_button.setDefaultAction(self.prevFrameAction)
        self.resegment_button = qw.QToolButton()
        self.resegment_button.setSizePolicy(qw.QSizePolicy.Minimum,
                                       qw.QSizePolicy.MinimumExpanding)
        self.resegment_button.setDefaultAction(self.resegmentAction)
        self.batchSegment_button = qw.QToolButton()
        self.batchSegment_button.setDefaultAction(self.batchSegmentAction)
        self.batchSegment_button.setSizePolicy(qw.QSizePolicy.Minimum,
                                       qw.QSizePolicy.MinimumExpanding)
        self.clear_cur_button = qw.QToolButton()
        self.clear_cur_button.setSizePolicy(qw.QSizePolicy.Minimum,
                                       qw.QSizePolicy.MinimumExpanding)
        self.clear_cur_button.setDefaultAction(self.clearCurrentAction)
        self.clear_all_button = qw.QToolButton()
        self.clear_all_button.setSizePolicy(qw.QSizePolicy.Minimum,
                                       qw.QSizePolicy.MinimumExpanding)
        self.clear_all_button.setDefaultAction(self.clearAllAction)
        # self.export_button = qw.QToolButton()
        # self.export_button.setDefaultAction(self.exportSegmentationAction)
        # layout.addWidget(self.export_button)
        self.keep_button = qw.QToolButton()
        self.keep_button.setSizePolicy(qw.QSizePolicy.Minimum,
                                       qw.QSizePolicy.MinimumExpanding)
        self.keep_button.setDefaultAction(
            self.display_widget.keepSelectedAction)
        self.remove_button = qw.QToolButton()
        self.remove_button.setSizePolicy(qw.QSizePolicy.Minimum,
                                         qw.QSizePolicy.MinimumExpanding)
        self.remove_button.setDefaultAction(
            self.display_widget.removeSelectedAction)

        layout = qw.QVBoxLayout()
        layout.addWidget(self.display_widget.seglist, 1)

        layout.addWidget(self.keep_button)
        layout.addWidget(self.remove_button)
        layout.addWidget(self.next_button)
        layout.addWidget(self.prev_button)
        layout.addWidget(self.resegment_button)
        layout.addWidget(self.clear_cur_button)
        layout.addWidget(self.clear_all_button)
        widget = qw.QWidget()
        widget.setLayout(layout)
        self.seg_dock = qw.QDockWidget('Segmented objects')
        self.seg_dock.setWidget(widget)
        self.seg_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                      qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.LeftDockWidgetArea, self.seg_dock)

    def _makeActions(self):
        self.imagedirAction = qw.QAction('Open image dir')
        self.imagedirAction.triggered.connect(self.openImageDir)
        self.outdirAction = qw.QAction('Open output directory')
        self.outdirAction.triggered.connect(self.setOutputDir)
        self.nextFrameAction = qw.QAction('&Next image (PgDn)')
        self.nextFrameAction.triggered.connect(self.nextFrame)
        self.prevFrameAction = qw.QAction('&Previous image (PgUp)')
        self.prevFrameAction.triggered.connect(self.prevFrame)
        self.resegmentAction = qw.QAction('Re-segment current image (R)')
        self.resegmentAction.triggered.connect(
            self.resegmentCurrent)
        self.batchSegmentAction = qw.QAction('Segment all files in directory')
        self.batchSegmentAction.triggered.connect(self.batchSegment)
        self.clearCurrentAction = qw.QAction('&Clear current segmentation (C)')
        self.clearCurrentAction.triggered.connect(self.clearCurrent)
        self.clearAllAction = qw.QAction('Reset all segmentation')
        self.clearAllAction.triggered.connect(self.clearAllSegmentation)
        self.exportSegmentationAction = qw.QAction(
            '&Export training and validation data (Ctrl+E)')
        self.exportSegmentationAction.triggered.connect(self.exportSegmentation)
        self.saveSegmentationAction = qw.QAction('&Save segmentations (Ctrl+S)')
        self.saveSegmentationAction.triggered.connect(self.saveSegmentation)
        self.loadSegmentationsAction = qw.QAction('&Open saved segmentations (Ctrl+O)')
        self.loadSegmentationsAction.triggered.connect(self.loadSegmentation)
        self.debugAction = qw.QAction('Debug')
        self.debugAction.setCheckable(True)
        v = settings.value('ytrainer/debug', logging.INFO)
        self.setDebug(v == logging.DEBUG)
        self.debugAction.setChecked(v == logging.DEBUG)
        self.debugAction.triggered.connect(self.setDebug)

    def _makeShortcuts(self):
        self.zoomInKey = qw.QShortcut(qg.QKeySequence('+'), self)
        self.zoomInKey.activated.connect(self.display_widget.zoomIn)
        self.zoomOutKey = qw.QShortcut(qg.QKeySequence('-'), self)
        self.zoomOutKey.activated.connect(self.display_widget.zoomOut)

        self.nextImageKey = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_PageDown), self)
        self.nextImageKey.activated.connect(self.nextFrame)
        self.prevImageKey = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_PageUp), self)
        self.prevImageKey.activated.connect(self.prevFrame)

        self.removeSegKey = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Delete), self)
        self.removeSegKey.activated.connect(
            self.display_widget.removeSelectedAction.trigger)
        self.removeSegKey2 = qw.QShortcut(qg.QKeySequence('X'), self)
        self.removeSegKey2.activated.connect(
            self.display_widget.removeSelectedAction.trigger)
        self.keepSegKey = qw.QShortcut(qg.QKeySequence('K'), self)
        self.keepSegKey.activated.connect(
            self.display_widget.keepSelectedAction.trigger)
        self.keepSegKey2 = qw.QShortcut(qg.QKeySequence('Shift+X'), self)
        self.keepSegKey2.activated.connect(
            self.display_widget.keepSelectedAction.trigger)

        self.clearCurrentImageKey = qw.QShortcut(qg.QKeySequence('C'), self)
        self.clearCurrentImageKey.activated.connect(self.clearCurrent)
        self.resegmentCUrrentImageKey = qw.QShortcut(qg.QKeySequence('R'), self)
        self.resegmentCUrrentImageKey.activated.connect(self.resegmentCurrent)

        self.saveKey = qw.QShortcut(qg.QKeySequence('Ctrl+S'), self)
        self.saveKey.activated.connect(
            self.saveSegmentation)
        self.openKey = qw.QShortcut(qg.QKeySequence('Ctrl+O'), self)
        self.openKey.activated.connect(
            self.loadSegmentation)

        self.exportKey = qw.QShortcut(qg.QKeySequence('Ctrl+E'), self)
        self.exportKey.activated.connect(self.exportSegmentation)

    def _makeMenuBar(self):
        self.fileMenu = self.menuBar().addMenu('&File')
        self.fileMenu.addActions([self.imagedirAction,
                                  self.outdirAction,
                                  self.loadSegmentationsAction,
                                  self.saveSegmentationAction,
                                  self.exportSegmentationAction])
        self.segMenu = self.menuBar().addMenu('&Segment')
        self.segMenu.addActions([self.nextFrameAction,
                                 self.prevFrameAction,
                                 self.resegmentAction,
                                 self.batchSegmentAction,
                                 self.clearCurrentAction,
                                 self.clearAllAction])
        self.viewMenu = self.menuBar().addMenu('View')
        self.viewMenu.addActions([self.display_widget.zoomInAction,
                                   self.display_widget.zoomOutAction,
                                   self.display_widget.autoColorAction,
                                   self.display_widget.colormapAction,
                                   self.display_widget.lineWidthAction])
        self.advancedMenu = self.menuBar().addMenu('Advanced')
        self.advancedMenu.addAction(self.debugAction)

    @qc.pyqtSlot(bool)
    def setDebug(self, val: bool):
        level = logging.DEBUG if val else logging.INFO
        logging.getLogger().setLevel(level)
        settings.setValue('ytrainer/debug', level)

    def outlineStyleToBoundaryMode(self, style):
        if style == OutlineStyle.bbox:
            self.sigSetDisplayGeom.emit(argos.constants.DrawingGeom.rectangle)
        else:
            self.sigSetDisplayGeom.emit(argos.constants.DrawingGeom.polygon)

    def openImageDir(self):
        directory = settings.value('training/imagedir', '.')
        directory = qw.QFileDialog.getExistingDirectory(self,
                                                        'Open image diretory',
                                                        directory=directory)
        logging.debug(f'Opening directory "{directory}"')
        if len(directory) == 0:
            return
        try:
            self.image_dir = directory
            self.image_dir_name.setText(directory)
            self.image_files = [entry.path for entry in
                                os.scandir(self.image_dir)]
            self.image_index = -1
            settings.setValue('training/imagedir', directory)
        except IOError as err:
            qw.QMessageBox.critical(self, 'Could not open image directory',
                                    str(err))
            return
        self.file_view.setRootIndex(self.file_model.setRootPath(self.image_dir))

    def setOutputDir(self):
        directory = settings.value('training/outdir', '.')
        directory = qw.QFileDialog.getExistingDirectory(self,
                                                        'Open image diretory',
                                                        directory=directory)
        logging.debug(f'Opening directory "{directory}"')
        if len(directory) == 0:
            return
        try:
            self.out_dir = directory
            self.out_dir_name.setText(directory)
            settings.setValue('training/outdir', directory)
        except IOError as err:
            qw.QMessageBox.critical(self,
                                    'Could create training/validation directory',
                                    str(err))
            return

    def gotoFrame(self, index):
        if index >= len(self.image_files) or index < 0 or self._waiting:
            return
        fname = self.image_files[index]
        if not os.path.exists(fname):
            qw.QMessageBox.critical(self, 'File does not exist', f'No such file exists: {fname}')
            del self.image_files[index]
            self.seg_dict.pop(index, None)
            return
        image = cv2.imread(fname)
        if image is None:
            return
        self.image_index = index
        self.display_widget.resetArenaAction.trigger()
        self.sigImage.emit(image, index)
        self.display_widget.updateSegList({})
        if fname not in self.seg_dict:
            self.saved = False
            self._waiting = True
            self.statusBar().showMessage(
                f'Processing image: {os.path.basename(fname)}.'
                f'[Image {self.image_index + 1} of {len(self.image_files)}] ...')            
            self.sigSegment.emit(image, index)
            
        else:
            self.sigSegmented.emit(self.seg_dict[fname], index)
            self.statusBar().showMessage(
                f'Current image: {os.path.basename(fname)}.'
                f'[Image {self.image_index + 1} of {len(self.image_files)}]')

    def nextFrame(self):
        self.gotoFrame(self.image_index + 1)

    def prevFrame(self):
        self.gotoFrame(self.image_index - 1)

    def handleFileSelectionChanged(self, selection):
        indices = selection.indexes()
        if len(indices) == 0:
            return
        fname = self.file_model.data(indices[0])
        index = self.image_files.index(os.path.join(self.image_dir, fname))
        self.gotoFrame(index)

    @qc.pyqtSlot(dict)
    def setSegmented(self, segdict: Dict[int, np.ndarray]) -> None:
        """Store the list of segmented objects for frame"""
        logging.debug(f'Received segmentated {len(segdict)} objects from {self.sender()}')
        fname = self.image_files[self.image_index]
        self.seg_dict[fname] = segdict
        self._waiting = False
        self.statusBar().showMessage(
            f'Current image: {os.path.basename(fname)}.'
            f'[Image {self.image_index + 1} of {len(self.image_files)}]')

    @qc.pyqtSlot(dict)
    def send_for_seg_and_wait(self, segdict: Dict[int, np.ndarray]) -> None:
        """Utility function for batch segmentation.

        When triggered send the next image file for processing
        """
        if len(segdict ) > 0:
            self.setSegmented(segdict)
        if len(self.seg_dict) == len(self.image_files):
            self.batch_indicator.setValue(self.batch_indicator.maximum() + 1)
            # Switch the connection back for interactive segmentation
            try:
                self.display_widget.sigPolygons.disconnect(
                    self.send_for_seg_and_wait)
            except TypeError:
                logging.error('Failed to disconnect: send_for_seg_and_wait')
            self.display_widget.sigPolygons.connect(
                self.setSegmented)
            return
        self.batch_indicator.setValue(self.image_index)
        self.gotoFrame(self.image_index + 1)        
        
    @qc.pyqtSlot()
    def batchSegment(self):
        """This works by switching the display_widget.sigPolygons from slot
        setSegmented to send_for_seg_and_wait.

        
        """
        maxcount = len(self.image_files)
        self.batch_indicator = qw.QProgressDialog('Processing all files in directory',
                                       None,
                                       0, maxcount,
                                       self)        
        self.batch_indicator.setWindowModality(qc.Qt.WindowModal)
        self.batch_indicator.show()
        try:
            self.display_widget.sigPolygons.disconnect(
                self.setSegmented)
        except TypeError:
            logging.error('Failed to disconnect: setSegmented')
        self.display_widget.sigPolygons.connect(
            self.send_for_seg_and_wait)
        self.image_index = -1
        self.send_for_seg_and_wait({})

    def cleanup(self):
        self.sigQuit.emit()
        settings.sync()
        logging.debug('Saved settings')

    def closeEvent(self, a0: qg.QCloseEvent) -> None:
        if self.saved:
            a0.accept()
        else:
            ret = qw.QMessageBox.question(self, 'Quit without saving?',
                                          'Data not saved. Are you sure to quit?'
                                          ' If not, select "No" and use the'
                                          ' "Export training/validation data"'
                                          ' button to save the data.',
                                          qw.QMessageBox.Yes,
                                          qw.QMessageBox.No)
            if ret == qw.QMessageBox.Yes:
                a0.accept()
            else:
                a0.ignore()

    def clearAllSegmentation(self):
        self.seg_dict = {}

    def resegmentCurrent(self):
        self.seg_dict.pop(self.image_files[self.image_index], None)
        self.gotoFrame(self.image_index)

    def clearCurrent(self):
        self.seg_dict.pop(self.image_files[self.image_index], None)
        self.display_widget.setPolygons({}, self.image_index)

    def _makeCocoDialog(self):
        dialog = qw.QDialog(self)
        layout = qw.QFormLayout()
        desc_label = qw.QLabel('Description')
        desc_text = qw.QLineEdit()
        desc_text.setText(self.description)
        def setDesc():
            self.description = desc_text.text()

        desc_text.editingFinished.connect(setDesc)
        license_label = qw.QLabel('License name')
        license_text = qw.QLineEdit()
        license_text.setText(self.license_name)
        def setLicenseName():
            self.license_name = license_text.text()

        license_text.editingFinished.connect(setLicenseName)
        layout.addRow(license_label, license_text)
        license_url_label = qw.QLabel('License URL')
        license_url_text = qw.QLineEdit()
        license_url_text.setText(self.license_url)
        def setLicenseUrl():
            self.license_url = license_url_text.text()

        license_url_text.editingFinished.connect(setLicenseUrl)
        layout.addRow(license_url_label, license_url_text)
        url_label = qw.QLabel('URL')
        url_text = qw.QLineEdit()

        def setUrl():
            self.url = url_text.text()

        url_text.editingFinished.connect(setUrl)
        layout.addRow(url_label, url_text)
        contrib_label = qw.QLabel('Contributor')
        contrib_text = qw.QLineEdit()

        def setContrib():
            self.contributor = contrib_text.text()

        contrib_text.editingFinished.connect(setContrib)
        layout.addRow(contrib_label, contrib_text)
        cat_label = qw.QLabel('Object class')
        cat_text = qw.QLineEdit(self.category_name)

        def setCategory():
            self.category_name = cat_text.text()

        cat_text.editingFinished.connect(setCategory)
        layout.addRow(cat_label, cat_text)

        size_label = qw.QLabel('Maximum image size')
        size_text = qw.QLabel(str(self.inputImageSize))
        layout.addRow(size_label, size_text)

        baseconfig_label = qw.QLabel('Neural-Net base configuration')
        baseconfig_combo = qw.QComboBox()
        for name in dir(yconfig):
            if name.startswith('yolact') and name.endswith('config'):
                baseconfig_combo.addItem(name)
        def setBaseconfig(text):
            self.baseconfig_name = text
            self.baseconfig = getattr(yconfig, text)
            size_text.setText(str(self.baseconfig.max_size))
        baseconfig_combo.currentTextChanged.connect(setBaseconfig)
        self.baseconfig_name = baseconfig_combo.currentText()
        layout.addRow(baseconfig_label, baseconfig_combo)

        val_label = qw.QLabel('Use % of images for validation')
        val_text = qw.QLineEdit(str(int(self.validation_frac * 100)))

        def setValFrac():
            self.validation_frac = float(val_text.text()) / 100

        val_text.editingFinished.connect(setValFrac)
        layout.addRow(val_label, val_text)

        subregion_label = qw.QLabel('Split into subregions')
        subregion_spin = qw.QSpinBox()
        subregion_spin.setRange(1, 5)
        subregion_spin.setValue(self.numCrops)
        def setSubregionCount(num):
            self.numCrops = num
        subregion_spin.valueChanged.connect(setSubregionCount)
        layout.addRow(subregion_label, subregion_spin)

        bbox_label = qw.QLabel('Export boundaries as')
        bbox_combo = qw.QComboBox()
        bbox_combo.addItems(['contour', 'bbox', 'minrect'])

        def setBoundaryType(text):
            self.boundaryType = text

        bbox_combo.currentTextChanged.connect(setBoundaryType)
        layout.addRow(bbox_label, bbox_combo)
        display_seg_button = qw.QCheckBox('Display segmentation (for debugging)')
        display_seg_button.setChecked(self.displayCoco)

        def setDisplayCocoSeg(state):
            self.displayCoco = state

        display_seg_button.clicked.connect(setDisplayCocoSeg)
        layout.addWidget(display_seg_button)

        ok_button = qw.QPushButton('OK')
        ok_button.setDefault(True)
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)
        dialog.setLayout(layout)
        ret = dialog.exec_()
        return ret


    def exportSegmentation(self):
        self.setOutputDir()
        train_dir = f'{self.out_dir}/training'
        try:
            os.mkdir(train_dir)
        except FileExistsError:
            qw.QMessageBox.critical(self, 'Directory already exists',
                                    f'Directory {train_dir} already exists.'
                                    f' Delete it or specify another output'
                                    f' directory')
            return
        except FileNotFoundError as ferr:
            qw.QMessageBox.critical(self, 'Path does not exist', str(ferr))
            return
        val_dir = f'{self.out_dir}/validation'
        try:
            os.mkdir(val_dir)
        except FileExistsError:
            qw.QMessageBox.critical('Directory already exists',
                                    f'Directory {val_dir} already exists.'
                                    f' Delete it or specify another output'
                                    f' directory')
            return
        ts = datetime.now()

        accepted = self._makeCocoDialog()
        validation_count = int(len(self.image_files) * self.validation_frac)
        training_count = len(self.image_files) - validation_count
        training_list = random.sample(self.image_files, training_count)
        self.dumpCocoJson(training_list, train_dir, ts,
                          message='Exporting training set in COCO format')
        yolact_config = {'name': f'{self.category_name}_weights',
                         'base': self.baseconfig_name,
                         'dataset': {'name': self.description,
                                     'train_info': f'{train_dir}/annotations.json',
                                     'valid_info': f'{val_dir}/annotations.json',
                                     'train_images': train_dir,
                                     'valid_images': val_dir,
                                     'has_gt': True,
                                     'class_names': [self.category_name]},
                         'num_classes': 2,
                         'max_size': self.inputImageSize,
                         'lr_steps': [100000, 150000, 175000, 190000],
                         'max_iter': 200000}
        yolact_file = f'{self.out_dir}/yolact_config.yaml'
        with open(yolact_file, 'w') as yolact_fd:
            yaml.dump(yolact_config, yolact_fd)
        if validation_count > 0:
            validation_list = set(self.image_files) - set(training_list)
            self.dumpCocoJson(validation_list, val_dir, ts,
                              message='Exporting validation set in COCO format')
        command = f'python -m yolact.train --config={yolact_file} --save_folder={self.out_dir}'
        qw.QMessageBox.information(self, 'Data saved',
                                   f'Training images: {train_dir}<br>'
                                   f'Validation images: {val_dir}<br>'
                                   f'Yolact configuration: {yolact_file}<br>'
                                   f'Now you can train yolact by running this command (copied to clipboard):<br>'
                                   f'<b>{command}</b><br>'
                                   f'But you must copy the initial weights file {self.baseconfig.backbone.path} to {self.out_dir} before starting<br>'
                                   f'For finer control over training settings see yolact help:'
                                   f'`python -m yolact.train --help`'
                                   )
        qw.qApp.clipboard().setText(command)

    def dumpCocoJson(self, filepaths, directory, ts, subregions=0,
                     message='Exporting COCO JSON'):
        """Dump annotation in COCO format as a .JSON file."""
        coco = {
            "info": {
                "description": self.description,
                "url": self.url,
                "version": '1.0',
                "year": ts.year,
                "contributor": self.contributor,
                "date_created": ts.isoformat(sep=' ')
            },
            "licenses": [
                {
                    "url": self.license_url,
                    "id": 0,
                    "name": self.license_name
                }
            ],
            'images': [],
            'type': 'instances',
            'annotations': [],
            'categories': [
                {'supercategory': None,
                 'id': 0,
                 'name': '_background_'},
                {'supercategory': None,
                 'id': 1,
                 'name': self.category_name}
            ]
        }
        imdir = os.path.join(directory, 'PNGImages')
        os.mkdir(imdir)
        seg_id = 0
        img_id = 0

        indicator = qw.QProgressDialog(message, None,
                                       0, len(filepaths),
                                       self)

        indicator.setWindowModality(qc.Qt.WindowModal)
        indicator.show()

        for ii, fpath in enumerate(filepaths):
            indicator.setValue(ii)
            if fpath not in self.seg_dict or len(self.seg_dict[fpath]) == 0:
                continue
            img = cv2.imread(fpath)
            fname = os.path.basename(fpath)
            prefix = fname.rpartition('.')[0]
            # If image is bigger than allowed size, make some random crops
            h = min(self.inputImageSize, img.shape[0])
            w = min(self.inputImageSize, img.shape[1])

            if img.shape[0] > self.inputImageSize or img.shape[1] > self.inputImageSize:
                # Here I select half of `num_crops` segments' top left corner (pos_tl)
                # and another half's bottom right corner.
                seg_bounds = [(np.min(seg[:, 0]), np.min(seg[:, 1]))
                              for seg in self.seg_dict[fpath].values()]
                seg_bounds = np.array(seg_bounds)
                idx = np.random.randint(0, len(seg_bounds), size=self.numCrops)
                xlist = seg_bounds[idx, 0] - np.random.randint(0, w // 2, size=len(idx))
                xlist[xlist < 0] = 0
                ylist = seg_bounds[idx, 1] - np.random.randint(0, h // 2, size=len(idx))
                ylist[ylist < 0] = 0
            else:
                xlist, ylist = [0], [0]
            for jj, (x, y) in enumerate(zip(xlist, ylist)):
                sq_img = np.zeros((self.max_size, self.max_size, 3),
                                  dtype=np.uint8)
                h_ = min(h, img.shape[0] - y)
                w_ = min(w, img.shape[1] - x)
                sq_img[:h_, :w_, :] = img[y: y + h_, x: x + w_, :]
                logging.debug(f'Processing: {prefix}: span ({x}, {y}, {x+h_}, {y+h_}')
                fname = f'{prefix}_{jj}.png'
                any_valid_seg = False
                for seg in self.seg_dict[fpath].values():
                    tmp_seg = seg - [x, y]
                    tmp_seg = tmp_seg[np.all((tmp_seg >= 0) &
                                             (tmp_seg < self.max_size),
                                             axis=1)]
                    if tmp_seg.shape[0] < 3:
                        continue
                    any_valid_seg = True
                    bbox = [int(xx) for xx in cv2.boundingRect(tmp_seg)]
                    if self.boundaryType == 'contour':
                        segmentation = [int(xx) for xx in tmp_seg.flatten()]
                    elif self.boundaryType == 'bbox':
                        segmentation = [bbox[0], bbox[1],
                                        bbox[0], bbox[1] + bbox[3],
                                        bbox[0] + bbox[2], bbox[1] + bbox[3],
                                        bbox[0] + bbox[2], bbox[1]]
                    elif self.boundaryType == 'minrect':
                        mr = cv2.minAreaRect(tmp_seg)
                        segmentation = [int(xx) for xx in cv2.boxPoints(mr)]
                    _seg = np.array(segmentation).reshape(-1, 2)
                    logging.debug(f'Segmentation: \n{_seg} \nafter translating \n{seg}\nto {x}, {y}')
                    if len(_seg) == 0:
                        logging.debug(f'Segmentation empty for ({x},{y}): {seg}')
                        continue
                    if self.displayCoco:
                        cv2.drawContours(sq_img, [_seg], -1, (0, 0, 255))
                        cv2.rectangle(sq_img, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 255))
                    annotation = {
                        "id": seg_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "segmentation": [segmentation],
                        "area": cv2.contourArea(tmp_seg),
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    coco['annotations'].append(annotation)
                    seg_id += 1
                if not any_valid_seg:
                    continue
                cv2.imwrite(os.path.join(imdir, fname),
                            sq_img)
                coco['images'].append({
                    "license": 0,
                    "url": None,
                    "file_name": f"PNGImages/{fname}",
                    "height": self.max_size,
                    "width": self.max_size,
                    "date_captured": None,
                    "id": img_id
                })
                if self.displayCoco:
                    winname = 'cvwin'
                    title = f'{fname}. Press `Esc` or `q` to hide. Any other key to fast forward.'
                    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(winname, 800, 600)
                    cv2.imshow(winname, sq_img)
                    cv2.setWindowTitle(winname, title)
                    key = cv2.waitKey(1000)
                    if key == 27 or key == ord('q'):
                        self.displayCoco = False
                        cv2.destroyAllWindows()
                img_id += 1
        with open(os.path.join(directory, 'annotations.json'), 'w') as fd:
            json.dump(coco, fd)
        cv2.destroyAllWindows()
        indicator.setValue(len(filepaths))

    def saveSegmentation(self):
        savedir = settings.value('training/savedir', '.')
        filename, _ = qw.QFileDialog.getSaveFileName(
            self,
            'Save current segmentation data',
            directory=savedir,
            filter='Pickle file (*.pkl *.pickle);;All files (*)')
        if len(filename) == 0:
            return
        data = {'image_dir': self.image_dir,
                'seg_dict': {fpath: seg for fpath, seg in self.seg_dict.items()}}
        with open(filename, 'wb') as fd:
            pickle.dump(data, fd)
        settings.setValue('training/savedir', os.path.dirname(filename))
        self.saved = True

    def loadSegmentation(self):
        savedir = settings.value('training/savedir', '.')
        filename, _ = qw.QFileDialog.getOpenFileName(
            self, 'Load saved segmentation', directory=savedir,
            filter='Pickle file (*.pkl *.pickle);;All files (*)')
        if len(filename) == 0:
            return
        with open(filename, 'rb') as fd:
            data = pickle.load(fd)
            self.image_dir = data['image_dir']
            seg_dict = data['seg_dict']
            self.image_files = [entry.path for entry in os.scandir(self.image_dir) if os.path.isfile(entry.path)]
            for key in list(seg_dict.keys()):
                if key not in self.image_files:
                    seg_dict.pop(key)
            self.seg_dict = {fpath: seg for fpath, seg in seg_dict.items()}
        settings.setValue('training/savedir', os.path.dirname(filename))
        self.gotoFrame(0)

if __name__ == '__main__':
    app = qw.QApplication(sys.argv)
    win = TrainingWidget()
    win.setMinimumSize(800, 600)
    win.setWindowTitle('Argos - generate training data')
    win.showMaximized()
    app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())
