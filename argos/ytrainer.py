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
        # self.scene().sigPolygons.connect(self.sigPolygons)
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
        self.licenseName = ''
        self.licenseUrl = ''
        self.contributor = ''
        self.categoryName = 'object'
        self.url = ''
        self.inputImageSize = 550
        self.imageDir = settings.value('training/imagedir', '.')
        self.imageFiles = []
        self.imageIndex = -1
        self.trainingDir = 'training'
        self.validation_dir = 'validation'
        self.outputDir = settings.value('training/outdir', '.')
        self.baseconfigName = ''
        for name in dir(yconfig):
            if name.startswith('yolact') and name.endswith('config'):
                self.baseconfigName = name
                break
        self.baseconfig = getattr(yconfig, self.baseconfigName)
        self.weightsFile = ''
        self.configFile = ''
        self.categoryName = 'object'
        self.segDict = {}  # dict containing segmentation info for each file
        self.segWidget = SegWidget()
        self.segWidget.outline_combo.setCurrentText('contour')
        self.segWidget.setOutlineStyle('contour')
        self.limitsWidget = LimitsWidget()
        self.segDock = qw.QDockWidget('Segmentation settings')
        self.segDock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                     qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self.segDock)
        layout = qw.QVBoxLayout()
        layout.addWidget(self.segWidget)
        layout.addWidget(self.limitsWidget)
        widget = qw.QWidget()
        widget.setLayout(layout)
        scroll = qw.QScrollArea()
        scroll.setWidget(widget)
        self.segDock.setWidget(scroll)
        self.displayWidget = SegDisplay()
        self.displayWidget.setRoiMode()
        self.displayWidget.frame_scene.linewidth = 0
        self.setCentralWidget(self.displayWidget)
        self._makeActions()
        self._makeFileDock()
        self._makeSegDock()
        self._makeMenuBar()
        self.sigImage.connect(self.displayWidget.setFrame)
        self.sigSegment.connect(self.segWidget.sigProcess)
        self.segWidget.sigSegPolygons.connect(
            self.displayWidget.sigSetPolygons)
        self.displayWidget.sigPolygons.connect(self.setSegmented)
        self.segWidget.sigProcessed.connect(self.displayWidget.setBboxes)
        self.limitsWidget.sigWmin.connect(self.segWidget.setWmin)
        self.limitsWidget.sigWmax.connect(self.segWidget.setWmax)
        self.limitsWidget.sigHmin.connect(self.segWidget.setHmin)
        self.limitsWidget.sigHmax.connect(self.segWidget.setHmax)
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
        #   `segDict`, `sigSegmented` sends segmented polygons to
        #   displaywidget's `setPolygons` slot directly
        self.sigSegmented.connect(self.displayWidget.setPolygons)
        self.sigQuit.connect(self.segWidget.sigQuit)
        self.sigQuit.connect(self.limitsWidget.sigQuit)
        self._makeShortcuts()
        self.openImageDir()
        self.statusBar().showMessage('Press `Next image` to start segmenting')

    def _makeFileDock(self):
        self.file_dock = qw.QDockWidget('Files/Dirs')

        dirlayout = qw.QFormLayout()
        self.out_dir_label = qw.QLabel('Output directory for training data')
        self.out_dir_name = qw.QLabel(self.outputDir)
        dirlayout.addRow(self.out_dir_label, self.out_dir_name)
        self.imageDirLabel = qw.QLabel('Input image directory')
        self.imageDirName = qw.QLabel(self.imageDir)
        dirlayout.addRow(self.imageDirLabel, self.imageDirName)
        self.dir_widget = qw.QWidget()
        self.dir_widget.setLayout(dirlayout)

        self.fileView = qw.QListView()
        self.fileView.setSizeAdjustPolicy(qw.QListWidget.AdjustToContents)
        self.file_model = qw.QFileSystemModel()
        self.file_model.setFilter(qc.QDir.NoDotAndDotDot | qc.QDir.Files)
        self.fileView.setModel(self.file_model)
        self.fileView.setRootIndex(self.file_model.setRootPath(self.imageDir))
        self.fileView.selectionModel().selectionChanged.connect(self.handleFileSelectionChanged)

        self.fwidget = qw.QWidget()
        layout = qw.QVBoxLayout()
        layout.addWidget(self.dir_widget)
        layout.addWidget(self.fileView)
        self.fwidget.setLayout(layout)
        self.file_dock.setWidget(self.fwidget)
        self.file_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                       qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self.file_dock)

    def _makeSegDock(self):
        self.nextButton = qw.QToolButton()
        self.nextButton.setSizePolicy(qw.QSizePolicy.Minimum,
                                      qw.QSizePolicy.MinimumExpanding)
        self.nextButton.setDefaultAction(self.nextFrameAction)

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
        self.keepButton = qw.QToolButton()
        self.keepButton.setSizePolicy(qw.QSizePolicy.Minimum,
                                      qw.QSizePolicy.MinimumExpanding)
        self.keepButton.setDefaultAction(
            self.displayWidget.keepSelectedAction)
        self.removeButton = qw.QToolButton()
        self.removeButton.setSizePolicy(qw.QSizePolicy.Minimum,
                                        qw.QSizePolicy.MinimumExpanding)
        self.removeButton.setDefaultAction(
            self.displayWidget.removeSelectedAction)

        layout = qw.QVBoxLayout()
        layout.addWidget(self.displayWidget.segList, 1)

        layout.addWidget(self.keepButton)
        layout.addWidget(self.removeButton)
        layout.addWidget(self.nextButton)
        layout.addWidget(self.prev_button)
        layout.addWidget(self.resegment_button)
        layout.addWidget(self.clear_cur_button)
        layout.addWidget(self.clear_all_button)
        widget = qw.QWidget()
        widget.setLayout(layout)
        self.segDock = qw.QDockWidget('Segmented objects')
        self.segDock.setWidget(widget)
        self.segDock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                     qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.LeftDockWidgetArea, self.segDock)

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
        self.zoomInKey.activated.connect(self.displayWidget.zoomIn)
        self.zoomOutKey = qw.QShortcut(qg.QKeySequence('-'), self)
        self.zoomOutKey.activated.connect(self.displayWidget.zoomOut)

        self.nextImageKey = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_PageDown), self)
        self.nextImageKey.activated.connect(self.nextFrame)
        self.prevImageKey = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_PageUp), self)
        self.prevImageKey.activated.connect(self.prevFrame)

        self.removeSegKey = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Delete), self)
        self.removeSegKey.activated.connect(
            self.displayWidget.removeSelectedAction.trigger)
        self.removeSegKey2 = qw.QShortcut(qg.QKeySequence('X'), self)
        self.removeSegKey2.activated.connect(
            self.displayWidget.removeSelectedAction.trigger)
        self.keepSegKey = qw.QShortcut(qg.QKeySequence('K'), self)
        self.keepSegKey.activated.connect(
            self.displayWidget.keepSelectedAction.trigger)
        self.keepSegKey2 = qw.QShortcut(qg.QKeySequence('Shift+X'), self)
        self.keepSegKey2.activated.connect(
            self.displayWidget.keepSelectedAction.trigger)

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
        self.viewMenu.addActions([self.displayWidget.zoomInAction,
                                  self.displayWidget.zoomOutAction,
                                  self.displayWidget.autoColorAction,
                                  self.displayWidget.colormapAction,
                                  self.displayWidget.lineWidthAction,
                                  self.displayWidget.fontSizeAction,
                                  self.displayWidget.relativeFontSizeAction])
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
            self.imageDir = directory
            self.imageDirName.setText(directory)
            self.imageFiles = [entry.path for entry in
                               os.scandir(self.imageDir)]
            self.imageIndex = -1
            settings.setValue('training/imagedir', directory)
        except IOError as err:
            qw.QMessageBox.critical(self, 'Could not open image directory',
                                    str(err))
            return
        self.fileView.setRootIndex(self.file_model.setRootPath(self.imageDir))

    def setOutputDir(self):
        directory = settings.value('training/outdir', '.')
        directory = qw.QFileDialog.getExistingDirectory(self,
                                                        'Open image diretory',
                                                        directory=directory)
        logging.debug(f'Opening directory "{directory}"')
        if len(directory) == 0:
            return
        try:
            self.outputDir = directory
            self.out_dir_name.setText(directory)
            settings.setValue('training/outdir', directory)
        except IOError as err:
            qw.QMessageBox.critical(self,
                                    'Could create training/validation directory',
                                    str(err))
            return

    def gotoFrame(self, index):
        if index >= len(self.imageFiles) or index < 0 or self._waiting:
            return
        fname = self.imageFiles[index]
        if not os.path.exists(fname):
            qw.QMessageBox.critical(self, 'File does not exist', f'No such file exists: {fname}')
            del self.imageFiles[index]
            self.segDict.pop(index, None)
            return
        image = cv2.imread(fname)
        if image is None:
            return
        self.imageIndex = index
        self.displayWidget.resetArenaAction.trigger()
        self.sigImage.emit(image, index)
        self.displayWidget.updateSegList({})
        if fname not in self.segDict:
            self.saved = False
            self._waiting = True
            self.statusBar().showMessage(
                f'Processing image: {os.path.basename(fname)}.'
                f'[Image {self.imageIndex + 1} of {len(self.imageFiles)}] ...')
            self.sigSegment.emit(image, index)
            print(f'#### Sent image: {os.path.basename(fname)}.'
            f'[Index {self.imageIndex} of {len(self.imageFiles)}]')
        else:
            self.sigSegmented.emit(self.segDict[fname], index)
            self.statusBar().showMessage(
                f'Current image: {os.path.basename(fname)}.'
                f'[Image {self.imageIndex + 1} of {len(self.imageFiles)}]')

    def nextFrame(self):
        self.gotoFrame(self.imageIndex + 1)

    def prevFrame(self):
        self.gotoFrame(self.imageIndex - 1)

    def handleFileSelectionChanged(self, selection):
        indices = selection.indexes()
        if len(indices) == 0:
            return
        fname = self.file_model.data(indices[0])
        index = self.imageFiles.index(os.path.join(self.imageDir, fname))
        self.gotoFrame(index)

    @qc.pyqtSlot(dict)
    def setSegmented(self, segdict: Dict[int, np.ndarray]) -> None:
        """Store the list of segmented objects for frame"""
        logging.debug(f'Received segmentated {len(segdict)} objects'
                      f' from {self.sender()} for image # {self.imageIndex}')

        fname = self.imageFiles[self.imageIndex]
        self.segDict[fname] = segdict
        self._waiting = False
        self.statusBar().showMessage(
            f'Current image: {os.path.basename(fname)}.'
            f' [Image {self.imageIndex + 1} of {len(self.imageFiles)}]')

    @qc.pyqtSlot(dict)
    def sendAndWaitSegmentation(self, segdict: Dict[int, np.ndarray]) -> None:
        """Utility function for batch segmentation.

        When triggered send the next image file for processing
        """
        if len(segdict ) > 0:
            self.setSegmented(segdict)
        # this comparison is needed because entries may be removed
        # from imageFiles in case of unreadable or deleted file
        if len(self.segDict) >= len(self.imageFiles):
            self.batchSegIndicator.setValue(self.batchSegIndicator.maximum())
            # Switch the connection back for interactive segmentation
            try:
                self.displayWidget.sigPolygons.disconnect(
                    self.sendAndWaitSegmentation)
            except TypeError:
                logging.error('Failed to disconnect: sendAndWaitSegmentation')
            self.displayWidget.sigPolygons.connect(
                self.setSegmented)
            return
        self.batchSegIndicator.setValue(self.imageIndex + 1)
        self.gotoFrame(self.imageIndex + 1)
        
    @qc.pyqtSlot()
    def batchSegment(self):
        """This works by switching the displayWidget.sigPolygons from slot
        setSegmented to sendAndWaitSegmentation.

        
        """
        maxcount = len(self.imageFiles)
        self.batchSegIndicator = qw.QProgressDialog('Processing all files in directory',
                                                    None,
                                                    0, maxcount,
                                                    self)
        self.batchSegIndicator.setWindowModality(qc.Qt.WindowModal)
        self.batchSegIndicator.setValue(0)
        self.batchSegIndicator.show()
        try:
            self.displayWidget.sigPolygons.disconnect(
                self.setSegmented)
        except TypeError:
            logging.error('Failed to disconnect: setSegmented')
        print('AAA. Polygon receivers', self.displayWidget.receivers(self.displayWidget.sigPolygons))
        self.displayWidget.sigPolygons.connect(
            self.sendAndWaitSegmentation)
        print('BBB. Polygon receivers',
              self.displayWidget.receivers(self.displayWidget.sigPolygons))
        self.imageIndex = -1
        self.sendAndWaitSegmentation({})

    def cleanup(self):
        self.sigQuit.emit()
        settings.sync()
        logging.debug('Saved settings')

    def closeEvent(self, a0: qg.QCloseEvent) -> None:
        if self.saved:
            a0.accept()
        else:
            ret = qw.QMessageBox.question(self, 'Quit without saving?',
                                          'Are you sure to quit?'
                                          ' Data not saved.'
                                          ' Select "No" and use the'
                                          ' "Export training/validation data"'
                                          ' button to save the data.',
                                          qw.QMessageBox.Yes,
                                          qw.QMessageBox.No)
            if ret == qw.QMessageBox.Yes:
                a0.accept()
            else:
                a0.ignore()

    def clearAllSegmentation(self):
        self.segDict = {}

    def resegmentCurrent(self):
        self.segDict.pop(self.imageFiles[self.imageIndex], None)
        self.gotoFrame(self.imageIndex)

    def clearCurrent(self):
        self.segDict.pop(self.imageFiles[self.imageIndex], None)
        self.displayWidget.setPolygons({}, self.imageIndex)

    def _makeCocoDialog(self):
        dialog = qw.QDialog(self)
        layout = qw.QFormLayout()
        descLabel = qw.QLabel('Description')
        descText = qw.QLineEdit()
        descText.setText(self.description)

        def setDesc():
            self.description = descText.text()

        descText.editingFinished.connect(setDesc)
        layout.addRow(descLabel, descText)
        licenseLabel = qw.QLabel('License name')
        licenseText = qw.QLineEdit()
        licenseText.setText(self.licenseName)

        def setLicenseName():
            self.licenseName = licenseText.text()

        licenseText.editingFinished.connect(setLicenseName)
        layout.addRow(licenseLabel, licenseText)
        licenseUrlLabel = qw.QLabel('License URL')
        licenseUrlText = qw.QLineEdit()
        licenseUrlText.setText(self.licenseUrl)

        def setLicenseUrl():
            self.licenseUrl = licenseUrlText.text()

        licenseUrlText.editingFinished.connect(setLicenseUrl)
        layout.addRow(licenseUrlLabel, licenseUrlText)
        urlLabel = qw.QLabel('URL')
        urlText = qw.QLineEdit()

        def setUrl():
            self.url = urlText.text()

        urlText.editingFinished.connect(setUrl)
        layout.addRow(urlLabel, urlText)
        contribLabel = qw.QLabel('Contributor')
        contribText = qw.QLineEdit()

        def setContrib():
            self.contributor = contribText.text()

        contribText.editingFinished.connect(setContrib)
        layout.addRow(contribLabel, contribText)
        catLabel = qw.QLabel('Object class')
        catText = qw.QLineEdit(self.categoryName)

        def setCategory():
            self.categoryName = catText.text()

        catText.editingFinished.connect(setCategory)
        layout.addRow(catLabel, catText)

        sizeLabel = qw.QLabel('Maximum image size')
        sizeText = qw.QLabel(str(self.inputImageSize))
        layout.addRow(sizeLabel, sizeText)

        baseConfigLabel = qw.QLabel('Neural-Net base configuration')
        baseConfigCombo = qw.QComboBox()
        for name in dir(yconfig):
            if name.startswith('yolact') and name.endswith('config'):
                baseConfigCombo.addItem(name)

        def setBaseconfig(text):
            self.baseconfigName = text
            self.baseconfig = getattr(yconfig, text)
            sizeText.setText(str(self.baseconfig.max_size))

        baseConfigCombo.currentTextChanged.connect(setBaseconfig)
        self.baseconfigName = baseConfigCombo.currentText()
        layout.addRow(baseConfigLabel, baseConfigCombo)

        valLabel = qw.QLabel('Use % of images for validation')
        valText = qw.QLineEdit(str(int(self.validation_frac * 100)))

        def setValFrac():
            self.validation_frac = float(valText.text()) / 100

        valText.editingFinished.connect(setValFrac)
        layout.addRow(valLabel, valText)

        subregionLabel = qw.QLabel('Split into subregions')
        subregionSpin = qw.QSpinBox()
        subregionSpin.setRange(1, 5)
        subregionSpin.setValue(self.numCrops)

        def setSubregionCount(num):
            self.numCrops = num

        subregionSpin.valueChanged.connect(setSubregionCount)
        layout.addRow(subregionLabel, subregionSpin)

        bboxLabel = qw.QLabel('Export boundaries as')
        bboxCombo = qw.QComboBox()
        bboxCombo.addItems(['contour', 'bbox', 'minrect'])

        def setBoundaryType(text):
            self.boundaryType = text

        bboxCombo.currentTextChanged.connect(setBoundaryType)
        layout.addRow(bboxLabel, bboxCombo)
        displaySegButton = qw.QCheckBox('Display segmentation (for debugging)')
        displaySegButton.setChecked(self.displayCoco)

        def setDisplayCocoSeg(state):
            self.displayCoco = state

        displaySegButton.clicked.connect(setDisplayCocoSeg)
        layout.addWidget(displaySegButton)

        okButton = qw.QPushButton('OK')
        okButton.setDefault(True)
        okButton.clicked.connect(dialog.accept)
        layout.addWidget(okButton)
        dialog.setLayout(layout)
        ret = dialog.exec_()
        return ret

    def exportSegmentation(self):
        self.setOutputDir()
        trainDir = f'{self.outputDir}/training'
        try:
            os.mkdir(trainDir)
        except FileExistsError:
            qw.QMessageBox.critical(self, 'Directory already exists',
                                    f'Directory {trainDir} already exists.'
                                    f' Delete it or specify another output'
                                    f' directory')
            return
        except FileNotFoundError as ferr:
            qw.QMessageBox.critical(self, 'Path does not exist', str(ferr))
            return
        valDir = f'{self.outputDir}/validation'
        try:
            os.mkdir(valDir)
        except FileExistsError:
            qw.QMessageBox.critical('Directory already exists',
                                    f'Directory {valDir} already exists.'
                                    f' Delete it or specify another output'
                                    f' directory')
            return
        ts = datetime.now()

        accepted = self._makeCocoDialog()
        validationCount = int(len(self.imageFiles) * self.validation_frac)
        trainingCount = len(self.imageFiles) - validationCount
        trainingList = random.sample(self.imageFiles, trainingCount)
        self.dumpCocoJson(trainingList, trainDir, ts,
                          message='Exporting training set in COCO format')
        yolactConfig = {'name': f'{self.categoryName}_weights',
                         'base': self.baseconfigName,
                         'dataset': {'name': self.description,
                                     'train_info': f'{trainDir}/annotations.json',
                                     'valid_info': f'{valDir}/annotations.json',
                                     'train_images': trainDir,
                                     'valid_images': valDir,
                                     'has_gt': True,
                                     'class_names': [self.categoryName]},
                         'num_classes': 2,
                         'max_size': self.inputImageSize,
                        'lr_steps': [100000, 150000, 175000, 190000],
                        'max_iter': 200000}
        yolactFile = f'{self.outputDir}/yolact_config.yaml'
        with open(yolactFile, 'w') as yolactFd:
            yaml.dump(yolactConfig, yolactFd)
        if validationCount > 0:
            validationList = set(self.imageFiles) - set(trainingList)
            self.dumpCocoJson(validationList, valDir, ts,
                              message='Exporting validation set in COCO format')
        command = f'python -m yolact.train --config={yolactFile} --save_folder={self.outputDir}'
        qw.QMessageBox.information(self, 'Data saved',
                                   f'Training images: {trainDir}<br>'
                                   f'Validation images: {valDir}<br>'
                                   f'Yolact configuration: {yolactFile}<br>'
                                   f'Now you can train yolact by running this command (copied to clipboard):<br>'
                                   f'<b>{command}</b><br>'
                                   f'But you must copy the initial weights file {self.baseconfig.backbone.path} to {self.outputDir} before starting<br>'
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
                    "url": self.licenseUrl,
                    "id": 0,
                    "name": self.licenseName
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
                 'name': self.categoryName}
            ]
        }
        imdir = os.path.join(directory, 'PNGImages')
        os.mkdir(imdir)
        segId = 0
        imgId = 0

        indicator = qw.QProgressDialog(message, None,
                                       0, len(filepaths),
                                       self)

        indicator.setWindowModality(qc.Qt.WindowModal)
        indicator.show()

        for ii, fpath in enumerate(filepaths):
            indicator.setValue(ii)
            if fpath not in self.segDict or len(self.segDict[fpath]) == 0:
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
                segBounds = [(np.min(seg[:, 0]), np.min(seg[:, 1]))
                             for seg in self.segDict[fpath].values()]
                segBounds = np.array(segBounds)
                idx = np.random.randint(0, len(segBounds), size=self.numCrops)
                xlist = segBounds[idx, 0] - np.random.randint(0, w // 2, size=len(idx))
                xlist[xlist < 0] = 0
                ylist = segBounds[idx, 1] - np.random.randint(0, h // 2, size=len(idx))
                ylist[ylist < 0] = 0
            else:
                xlist, ylist = [0], [0]
            for jj, (x, y) in enumerate(zip(xlist, ylist)):
                sqImg = np.zeros((self.inputImageSize, self.inputImageSize, 3),
                                 dtype=np.uint8)
                h_ = min(h, img.shape[0] - y)
                w_ = min(w, img.shape[1] - x)
                sqImg[:h_, :w_, :] = img[y: y + h_, x: x + w_, :]
                logging.debug(f'Processing: {prefix}: span ({x}, {y}, {x+h_}, {y+h_}')
                fname = f'{prefix}_{jj}.png'
                anyValidSeg = False
                for seg in self.segDict[fpath].values():
                    tmpSeg = seg - [x, y]
                    tmpSeg = tmpSeg[np.all((tmpSeg >= 0) &
                                           (tmpSeg < self.inputImageSize),
                                           axis=1)]
                    if tmpSeg.shape[0] < 3:
                        continue
                    anyValidSeg = True
                    bbox = [int(xx) for xx in cv2.boundingRect(tmpSeg)]
                    if self.boundaryType == 'contour':
                        segmentation = [int(xx) for xx in tmpSeg.flatten()]
                    elif self.boundaryType == 'bbox':
                        segmentation = [bbox[0], bbox[1],
                                        bbox[0], bbox[1] + bbox[3],
                                        bbox[0] + bbox[2], bbox[1] + bbox[3],
                                        bbox[0] + bbox[2], bbox[1]]
                    elif self.boundaryType == 'minrect':
                        mr = cv2.minAreaRect(tmpSeg)
                        segmentation = [int(xx) for xx in cv2.boxPoints(mr)]
                    _seg = np.array(segmentation).reshape(-1, 2)
                    logging.debug(f'Segmentation: \n{_seg} \nafter translating \n{seg}\nto {x}, {y}')
                    if len(_seg) == 0:
                        logging.debug(f'Segmentation empty for ({x},{y}): {seg}')
                        continue
                    if self.displayCoco:
                        cv2.drawContours(sqImg, [_seg], -1, (0, 0, 255))
                        cv2.rectangle(sqImg, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 255))
                    annotation = {
                        "id": segId,
                        "image_id": imgId,
                        "category_id": 1,
                        "segmentation": [segmentation],
                        "area": cv2.contourArea(tmpSeg),
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    coco['annotations'].append(annotation)
                    segId += 1
                if not anyValidSeg:
                    continue
                cv2.imwrite(os.path.join(imdir, fname),
                            sqImg)
                coco['images'].append({
                    "license": 0,
                    "url": None,
                    "file_name": f"PNGImages/{fname}",
                    "height": self.inputImageSize,
                    "width": self.inputImageSize,
                    "date_captured": None,
                    "id": imgId
                })
                if self.displayCoco:
                    winname = 'cvwin'
                    title = f'{fname}. Press `Esc` or `q` to hide. '  \
                            f'Any other key to fast forward.'
                    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(winname, 800, 600)
                    cv2.imshow(winname, sqImg)
                    cv2.setWindowTitle(winname, title)
                    key = cv2.waitKey(1000)
                    if key == 27 or key == ord('q'):
                        self.displayCoco = False
                        cv2.destroyAllWindows()
                imgId += 1
        with open(os.path.join(directory, 'annotations.json'), 'w') as fd:
            json.dump(coco, fd)
        cv2.destroyAllWindows()
        indicator.setValue(len(filepaths))

    def saveSegmentation(self):
        saveDir = settings.value('training/savedir', '.')
        filename, _ = qw.QFileDialog.getSaveFileName(
            self,
            'Save current segmentation data',
            directory=saveDir,
            filter='Pickle file (*.pkl *.pickle);;All files (*)')
        if len(filename) == 0:
            return
        data = {'image_dir': self.imageDir,
                'seg_dict': {fpath: seg for fpath, seg in self.segDict.items()}}
        with open(filename, 'wb') as fd:
            pickle.dump(data, fd)
        settings.setValue('training/savedir', os.path.dirname(filename))
        self.saved = True

    def loadSegmentation(self):
        saveDir = settings.value('training/savedir', '.')
        filename, _ = qw.QFileDialog.getOpenFileName(
            self, 'Load saved segmentation', directory=saveDir,
            filter='Pickle file (*.pkl *.pickle);;All files (*)')
        if len(filename) == 0:
            return
        with open(filename, 'rb') as fd:
            data = pickle.load(fd)
            self.imageDir = data['image_dir']
            segDict = data['seg_dict']
            self.imageFiles = [entry.path for entry in os.scandir(self.imageDir) if os.path.isfile(entry.path)]
            for key in list(segDict.keys()):
                if key not in self.imageFiles:
                    segDict.pop(key)
            self.segDict = {fpath: seg for fpath, seg in segDict.items()}
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
