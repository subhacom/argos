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

from argos import utility as ut
from argos import display
from argos.display import Display
from argos.segwidget import SegWidget
from yolact import config as yconfig

settings = ut.init()


class SegDisplay(Display):
    sigItemSelectionChanged = qc.pyqtSignal(list)
    sigPolygons = qc.pyqtSignal(dict)

    def __init__(self, *args, **kwargs):
        super(SegDisplay, self).__init__(*args, **kwargs)
        self.seglist = qw.QListWidget()
        self.seglist.setSizeAdjustPolicy(qw.QListWidget.AdjustToContents)
        self.seglist.setSelectionMode(self.seglist.ExtendedSelection)
        self.seglist.itemSelectionChanged.connect(self.sendSelection)
        self.sigItemSelectionChanged.connect(self.scene().setSelected)
        self.keepSelectedAction = qw.QAction('Keep selected objects')
        self.removeSelectedAction = qw.QAction('Remove selected objects')
        self.keepSelectedAction.triggered.connect(self.scene().keepSelected)
        self.removeSelectedAction.triggered.connect(self.scene().removeSelected)
        self.scene().sigPolygons.connect(self.sigPolygons)
        self.scene().sigPolygons.connect(self.updateSegList)

    @qc.pyqtSlot()
    def sendSelection(self):
        selection = [int(item.text()) for item in
                     self.seglist.selectedItems()]
        self.sigItemSelectionChanged.emit(selection)

    @qc.pyqtSlot(dict)
    def updateSegList(self, segdict: Dict[int, np.ndarray]) -> None:
        self.seglist.clear()
        self.seglist.addItems([str(key) for key in segdict.keys()])
        self.seglist.updateGeometry()

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
    sigSetDisplayGeom = qc.pyqtSignal(display.DrawingGeom)

    def __init__(self, *args, **kwargs):
        super(TrainingWidget, self).__init__(*args, **kwargs)
        self._waiting = False
        self.boundary_type = 'contour'
        self.display_coco = True
        self.num_crops = 4  # number of random crops to generate if input image is bigger than training image size
        self.saved = True
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
        self.seg_dock = qw.QDockWidget('Segmentation settings')
        self.seg_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                      qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self.seg_dock)
        scroll = qw.QScrollArea()
        scroll.setWidget(self.seg_widget)
        self.seg_dock.setWidget(scroll)
        self.display_widget = SegDisplay()
        self.display_widget.setRoiMode()
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
        self._makeShortcuts()
        self.openImageDir()
        self.statusBar().showMessage('Press `Next image` to start segmenting')

    def _makeFileDock(self):
        self.file_dock = qw.QDockWidget('Files/Dirs')
        layout = qw.QFormLayout()
        self.out_dir_label = qw.QLabel('Output directory for training data')
        self.out_dir_name = qw.QLabel(self.out_dir)
        layout.addRow(self.out_dir_label, self.out_dir_name)
        self.image_dir_label = qw.QLabel('Input image directory')
        self.image_dir_name = qw.QLabel(self.image_dir)
        layout.addRow(self.image_dir_label, self.image_dir_name)
        self.dir_widget = qw.QWidget()
        self.dir_widget.setLayout(layout)
        self.file_view = qw.QListView()
        self.file_view.setSizeAdjustPolicy(qw.QListWidget.AdjustToContents)
        self.file_model = qw.QFileSystemModel()
        self.file_model.setFilter(qc.QDir.NoDotAndDotDot | qc.QDir.Files)
        self.file_view.setModel(self.file_model)
        self.file_view.setRootIndex(self.file_model.setRootPath(self.image_dir))
        self.file_view.selectionModel().selectionChanged.connect(self.handleFileSelectionChanged)
        layout = qw.QVBoxLayout()
        layout.addWidget(self.dir_widget)
        layout.addWidget(self.file_view)
        self.fwidget = qw.QWidget()
        self.fwidget.setLayout(layout)
        scroll = qw.QScrollArea()
        scroll.setWidget(self.fwidget)
        self.file_dock.setWidget(scroll)
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
        self.nextFrameAction = qw.QAction('Next image')
        self.nextFrameAction.triggered.connect(self.nextFrame)
        self.prevFrameAction = qw.QAction('Previous image')
        self.prevFrameAction.triggered.connect(self.prevFrame)
        self.resegmentAction = qw.QAction('Re-segment current image')
        self.resegmentAction.triggered.connect(
            self.resegmentCurrent)
        self.clearCurrentAction = qw.QAction('Clear current segmentation')
        self.clearCurrentAction.triggered.connect(self.clearCurrent)
        self.clearAllAction = qw.QAction('Reset all segmentation')
        self.clearAllAction.triggered.connect(self.clearAllSegmentation)
        self.exportSegmentationAction = qw.QAction(
            'Export training/validation data')
        self.exportSegmentationAction.triggered.connect(self.exportSegmentation)
        self.saveSegmentationAction = qw.QAction('Save segmentations')
        self.saveSegmentationAction.triggered.connect(self.saveSegmentation)
        self.loadSegmentationsAction = qw.QAction('Load segmentations')
        self.loadSegmentationsAction.triggered.connect(self.loadSegmentation)

    def _makeShortcuts(self):
        self.sc_next = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_PageDown), self)
        self.sc_next.activated.connect(self.nextFrame)
        self.sc_prev = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_PageUp), self)
        self.sc_prev.activated.connect(self.prevFrame)
        self.sc_remove = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Delete), self)
        self.sc_remove.activated.connect(
            self.display_widget.removeSelectedAction.trigger)
        self.sc_remove_2 = qw.QShortcut(qg.QKeySequence('X'), self)
        self.sc_remove_2.activated.connect(
            self.display_widget.removeSelectedAction.trigger)
        self.sc_keep = qw.QShortcut(qg.QKeySequence('K'), self)
        self.sc_keep.activated.connect(
            self.display_widget.keepSelectedAction.trigger)
        self.sc_keep_2 = qw.QShortcut(qg.QKeySequence('Shift+X'), self)
        self.sc_keep_2.activated.connect(
            self.display_widget.keepSelectedAction.trigger)
        self.sc_save = qw.QShortcut(qg.QKeySequence('Ctrl+S'), self)
        self.sc_save.activated.connect(
            self.saveSegmentation)
        self.sc_open = qw.QShortcut(qg.QKeySequence('Ctrl+O'), self)
        self.sc_save.activated.connect(
            self.loadSegmentation)
        self.sc_export = qw.QShortcut(qg.QKeySequence('Ctrl+E'), self)
        self.sc_export.activated.connect(self.exportSegmentation)

    def _makeMenuBar(self):
        self.file_menu = self.menuBar().addMenu('&File')
        self.file_menu.addActions([self.imagedirAction,
                                   self.outdirAction,
                                   self.loadSegmentationsAction,
                                   self.saveSegmentationAction,
                                   self.exportSegmentationAction])
        self.seg_menu = self.menuBar().addMenu('&Segment')
        self.seg_menu.addActions([self.nextFrameAction,
                                  self.prevFrameAction,
                                  self.resegmentAction,
                                  self.clearCurrentAction,
                                  self.clearAllAction])
        self.view_menu = self.menuBar().addMenu('View')
        self.view_menu.addActions([self.display_widget.zoomInAction,
                                   self.display_widget.zoomOutAction])

    def outlineStyleToBoundaryMode(self, style):
        if style == ut.OutlineStyle.bbox:
            self.sigSetDisplayGeom.emit(display.DrawingGeom.rectangle)
        else:
            self.sigSetDisplayGeom.emit(display.DrawingGeom.polygon)

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
        self.image_index = index
        fname = self.image_files[index]
        image = cv2.imread(fname)
        self.display_widget.resetArenaAction.trigger()
        self.sigImage.emit(image, index)
        if index not in self.seg_dict:
            self.saved = False
            self.sigSegment.emit(image, index)
            self._waiting = True
        else:
            self.sigSegmented.emit(self.seg_dict[index], index)
        self.statusBar().showMessage(
            f'Current image: {os.path.basename(fname)}. [Image {self.image_index + 1} of {len(self.image_files)}]')

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
        logging.debug(f'Received segmentation {segdict} from {self.sender()}')
        self.seg_dict[self.image_index] = segdict
        self._waiting = False

    def cleanup(self):
        self.sigQuit.emit()
        settings.sync()
        logging.debug('Saved settings')

    def closeEvent(self, a0: qg.QCloseEvent) -> None:
        if self.saved:
            a0.accept()
        else:
            ret = qw.QMessageBox.question(self, 'Save segmented data',
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
        self.seg_dict.pop(self.image_index, None)
        self.gotoFrame(self.image_index)

    def clearCurrent(self):
        self.seg_dict.pop(self.image_index, None)
        self.display_widget.setPolygons({}, self.image_index)

    def _makeCocoDialog(self):
        dialog = qw.QDialog(self)
        layout = qw.QFormLayout()
        desc_label = qw.QLabel('Description')
        desc_text = qw.QLineEdit()

        def setDesc():
            self.description = desc_text.text()

        desc_text.editingFinished.connect(setDesc)
        license_label = qw.QLabel('License name')
        license_text = qw.QLineEdit()

        def setLicenseName():
            self.license_name = license_text.text()

        license_text.editingFinished.connect(setLicenseName)
        layout.addRow(license_label, license_text)
        license_url_label = qw.QLabel('License URL')
        license_url_text = qw.QLineEdit()

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
        size_text = qw.QLabel(str(self.max_size))
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

        bbox_label = qw.QLabel('Export boundaries as')
        bbox_combo = qw.QComboBox()
        bbox_combo.addItems(['contour', 'bbox', 'minrect'])
        def setBoundaryType(text):
            self.boundary_type = text
        bbox_combo.currentTextChanged.connect(setBoundaryType)
        layout.addRow(bbox_label, bbox_combo)
        display_seg_button = qw.QCheckBox('Display segmentation (for debugging)')
        display_seg_button.setChecked(self.display_coco)
        def setDisplayCocoSeg(state):
            self.display_coco = state
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
        self.validation_frac = 0.3
        self.description = None
        self.license_name = None
        self.license_url = None
        self.contributor = None
        self.category_name = 'object'
        self.url = None
        self.max_size = 550
        accepted = self._makeCocoDialog()
        validation_count = int(len(self.image_files) * self.validation_frac)
        training_count = len(self.image_files) - validation_count
        training_list = random.sample(self.image_files, training_count)
        self.dumpCocoJson(training_list, train_dir, ts)
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
                         'max_size': self.max_size,
                         'lr_steps': [100000, 150000, 175000, 190000],
                         'max_iter': 200000}
        yolact_file = f'{self.out_dir}/yolact_config.yaml'
        with open(yolact_file, 'w') as yolact_fd:
            yaml.dump(yolact_config, yolact_fd)
        if validation_count > 0:
            validation_list = random.sample(self.image_files, validation_count)
            self.dumpCocoJson(validation_list, val_dir, ts)
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

    def dumpCocoJson(self, filepaths, directory, ts):
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
            'categoeroes': [
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
        for fpath in filepaths:
            findex = self.image_files.index(fpath)
            if findex not in self.seg_dict or len(self.seg_dict[findex]) == 0:
                continue
            img = cv2.imread(fpath)
            fname = os.path.basename(fpath)
            prefix = fname.rpartition('.')[0]
            # If image is bigger than allowed size, make some random crops
            xlist = [0]
            ylist = [0]
            if img.shape[0] > self.max_size:
                if img.shape[0] >= self.max_size * 1.5:
                    ylist += random.sample(range(img.shape[0] - self.max_size),
                                          self.num_crops)
                else:
                    ylist += [img.shape[0] - self.max_size]
            if img.shape[1] > self.max_size:
                if img.shape[1] >= self.max_size * 1.5:
                    xlist += random.sample(range(img.shape[1] - self.max_size),
                                          self.num_crops)
                else:
                    xlist += [img.shape[1] - self.max_size]
            h = min(self.max_size, img.shape[0])
            w = min(self.max_size, img.shape[1])
            for jj, (x, y) in enumerate(zip(xlist, ylist)):
                sq_img = np.zeros((self.max_size, self.max_size, 3),
                                  dtype=np.uint8)
                sq_img[:h, :w, :] = img[y: y + h, x: x + w, :]
                logging.debug(f'Processing: {prefix}: span ({x}, {y}, {x+h}, {y+h}')
                fname = f'{prefix}_{jj}.png'
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

                for seg in self.seg_dict[findex].values():
                    tmp_seg = seg - [x, y]
                    tmp_seg = tmp_seg[np.all((tmp_seg >= 0) &
                                             (tmp_seg < self.max_size),
                                             axis=1)]
                    if tmp_seg.shape[0] < 3:
                        continue
                    bbox = [int(xx) for xx in cv2.boundingRect(tmp_seg)]
                    if self.boundary_type == 'contour':
                        segmentation = [int(xx) for xx in tmp_seg.flatten()]
                    elif self.boundary_type == 'bbox':
                        segmentation = [bbox[0], bbox[1],
                                        bbox[0], bbox[1] + bbox[3],
                                        bbox[0] + bbox[2], bbox[1] + bbox[3],
                                        bbox[0] + bbox[2], bbox[1]]
                    elif self.boundary_type == 'minrect':
                        mr = cv2.minAreaRect(tmp_seg)
                        segmentation = [int(xx) for xx in cv2.boxPoints(mr)]
                    _seg = np.array(segmentation).reshape(-1, 2)
                    logging.debug(f'Segmentation: \n{_seg} \nafter translating \n{seg}\nto {x}, {y}')
                    if len(_seg) == 0:
                        logging.debug(f'Segmentation empty for ({x},{y}): {seg}')
                        continue
                    if self.display_coco:
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
                if self.display_coco:
                    label = f'COCO export {fname}'
                    cv2.imshow(label, sq_img)
                    key = cv2.waitKey(1000)
                    if key == 27 or key == ord('q'):
                        self.display_coco = False
                    cv2.destroyAllWindows()

                img_id += 1
        with open(os.path.join(directory, 'annotations.json'), 'w') as fd:
            json.dump(coco, fd)

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
                'seg_dict': {self.image_files[index]: seg for index, seg in self.seg_dict.items()}}
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
            self.image_files = [entry.path for entry in os.scandir(self.image_dir)]
            for key in list(seg_dict.keys()):
                if key not in self.image_files:
                    seg_dict.pop(key)
            self.seg_dict = {self.image_files.index(key): seg for key, seg in seg_dict.items()}
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
