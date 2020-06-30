# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-29 4:52 PM
"""Widget to generate training data for YOLACT"""
import sys
import logging
import os
import numpy as np
import cv2
from typing import Dict
from PyQt5 import (
    QtCore as qc,
    QtWidgets as qw)

from argos import utility as ut
from argos.display import Display
from argos.segwidget import SegWidget


settings = ut.init()

class SegDisplay(Display):
    sigItemSelectionChanged = qc.pyqtSignal(list)
    sigPolygons = qc.pyqtSignal(dict)

    def __init__(self, *args, **kwargs):
        super(SegDisplay, self).__init__(*args, **kwargs)
        self.seglist = qw.QListWidget()
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


class TrainingWidget(qw.QMainWindow):
    sigQuit = qc.pyqtSignal()
    # Send an image and its index in file list for segmentation
    sigSegment = qc.pyqtSignal(np.ndarray, int)
    # Send the image
    sigImage = qc.pyqtSignal(np.ndarray, int)
    # send refined segmentation data
    sigSegmented = qc.pyqtSignal(dict, int)

    def __init__(self, *args, **kwargs):
        super(TrainingWidget, self).__init__(*args, **kwargs)
        self._waiting = False
        self.image_dir = settings.value('training/imagedir', '.')
        self.image_files = []
        self.image_index = -1
        self.training_dir = 'training'
        self.validation_dir = 'validation'
        self.out_dir = settings.value('training/outdir', '.')
        self.weights_file = ''
        self.config_file = ''
        self.seg_dict = {}   # dict containing segmentation info for each file
        self.seg_widget = SegWidget()
        self.seg_widget.setOutlineStyle(ut.OutlineStyle.contour)
        self.seg_dock = qw.QDockWidget('Segmentation settings')
        self.seg_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                       qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self.seg_dock)
        self.seg_dock.setWidget(self.seg_widget)
        self.display_widget = SegDisplay()
        self.seg_widget.sigSegPolygons.connect(self.display_widget.sigSetPolygons)
        self.display_widget.sigPolygons.connect(self.setSegmented)
        # self.display_widget.scene().setRoiRectMode()
        self.setCentralWidget(self.display_widget)
        self._makeActions()
        self._makeFileDock()
        self._makeSegDock()
        self._makeMenuBar()
        self.sigImage.connect(self.display_widget.setFrame)
        self.sigSegment.connect(self.seg_widget.sigProcess)
        self.seg_widget.sigProcessed.connect(self.display_widget.setPolygons)
        self.sigSegmented.connect(self.display_widget.setPolygons)
        self.sigQuit.connect(self.seg_widget.sigQuit)
        self.openImageDir()

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
        self.file_view  = qw.QListView()
        self.file_model = qw.QFileSystemModel()
        self.file_model.setFilter(qc.QDir.NoDotAndDotDot | qc.QDir.Files)
        self.file_view.setModel(self.file_model)
        self.file_view.setRootIndex(self.file_model.setRootPath(self.image_dir))
        layout = qw.QVBoxLayout()
        layout.addWidget(self.dir_widget)
        layout.addWidget(self.file_view)
        self.fwidget = qw.QWidget()
        self.fwidget.setLayout(layout)
        self.file_dock.setWidget(self.fwidget)
        self.file_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                       qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self.file_dock)

    def _makeSegDock(self):
        layout = qw.QVBoxLayout()
        layout.addWidget(self.display_widget.seglist)
        self.keep_button = qw.QToolButton()
        self.keep_button.setDefaultAction(self.display_widget.keepSelectedAction)
        layout.addWidget(self.keep_button)
        self.remove_button = qw.QToolButton()
        self.remove_button.setDefaultAction(self.display_widget.removeSelectedAction)
        layout.addWidget(self.remove_button)
        self.next_button = qw.QToolButton()
        self.next_button.setDefaultAction(self.nextFrameAction)
        layout.addWidget(self.next_button)
        self.prev_button = qw.QToolButton()
        self.prev_button.setDefaultAction(self.prevFrameAction)
        layout.addWidget(self.prev_button)
        self.clear_button = qw.QToolButton()
        self.clear_button.setDefaultAction(self.clearSegmentationAction)
        layout.addWidget(self.clear_button)
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
        self.clearSegmentationAction = qw.QAction('Reset segmentation')
        self.clearSegmentationAction.triggered.connect(self.clearSegmentation)

    def _makeMenuBar(self):
        self.file_menu = self.menuBar().addMenu('&File')
        self.file_menu.addAction(self.imagedirAction)
        self.file_menu.addAction(self.outdirAction)

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
            self.image_files = [entry.path for entry in os.scandir(self.image_dir)]
            self.image_index = -1
            settings.setValue('training/imagedir', directory)
        except IOError as err:
            qw.QMessageBox.critical(self, 'Could not open image directory', str(err))
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
        self.sigImage.emit(image, index)
        if index not in self.seg_dict:
            self.sigSegment.emit(image, index)
            self._waiting = True
        else:
            self.sigSegmented.emit(self.seg_dict[index], index)

    def nextFrame(self):
        self.gotoFrame(self.image_index + 1)

    def prevFrame(self):
        self.gotoFrame(self.image_index - 1)

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

    def clearSegmentation(self):
        self.seg_dict = {}

if __name__ == '__main__':
    app = qw.QApplication(sys.argv)
    win = TrainingWidget()
    win.setMinimumSize(800, 600)
    win.setWindowTitle('Argos - generate training data')
    win.showMaximized()
    app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())


