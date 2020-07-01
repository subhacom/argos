# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-29 4:52 PM
"""Widget to generate training data for YOLACT"""
import sys
import logging
import os
import random
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

    def setRoiMode(self):
        self.scene().setRoiPolygonMode()


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
        self.saved = True
        self.image_dir = settings.value('training/imagedir', '.')
        self.image_files = []
        self.image_index = -1
        self.training_dir = 'training'
        self.validation_dir = 'validation'
        self.out_dir = settings.value('training/outdir', '.')
        self.weights_file = ''
        self.config_file = ''
        self.category_name = 'object'
        self.seg_dict = {}  # dict containing segmentation info for each file
        self.seg_widget = SegWidget()
        self.seg_widget.setOutlineStyle(ut.OutlineStyle.contour)
        self.seg_dock = qw.QDockWidget('Segmentation settings')
        self.seg_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                      qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self.seg_dock)
        self.seg_dock.setWidget(self.seg_widget)
        self.display_widget = SegDisplay()
        self.display_widget.setRoiMode()
        self.seg_widget.sigSegPolygons.connect(
            self.display_widget.sigSetPolygons)
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
        self.file_view = qw.QListView()
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
        self.keep_button.setDefaultAction(
            self.display_widget.keepSelectedAction)
        layout.addWidget(self.keep_button)
        self.remove_button = qw.QToolButton()
        self.remove_button.setDefaultAction(
            self.display_widget.removeSelectedAction)
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
        self.export_button = qw.QToolButton()
        self.export_button.setDefaultAction(self.exportSegmentationAction)
        layout.addWidget(self.export_button)
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
        self.exportSegmentationAction = qw.QAction(
            'Export training/validation data')
        self.exportSegmentationAction.triggered.connect(self.exportSegmentation)

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
        self.sigImage.emit(image, index)
        if index not in self.seg_dict:
            self.saved = False
            self.sigSegment.emit(image, index)
            self._waiting = True
        else:
            self.sigSegmented.emit(self.seg_dict[index], index)
        self.statusBar().showMessage(
            f'Current image: {os.path.basename(fname)}')

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

    def clearSegmentation(self):
        self.seg_dict = {}

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
        size_text = qw.QLineEdit(str(self.max_size))

        def setMaxSize():
            self.max_size = int(size_text.text())

        size_text.editingFinished.connect(setMaxSize)
        layout.addRow(size_label, size_text)
        val_label = qw.QLabel('Use % of images for validation')
        val_text = qw.QLineEdit(str(int(self.validation_frac * 100)))

        def setValFrac():
            self.validation_frac = float(val_text.text()) / 100

        val_text.editingFinished.connect(setValFrac)
        layout.addRow(val_label, val_text)
        ok_button = qw.QPushButton('OK')
        ok_button.setDefault(True)
        layout.addWidget(ok_button)
        dialog.setLayout(layout)
        ret = dialog.result()
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
                         'dataset': {'name': self.description,
                                     'train_info': f'{train_dir}/annotations.json',
                                     'valid_info': f'{val_dir}/annotations.json',
                                     'train_images': train_dir,
                                     'valid_images': val_dir,
                                     'has_gt': True,
                                     'class_names': [self.category_name]},
                         'num_classes': 2,
                         'max_size': self.max_size,
                         'lr_steps': [28000, 60000, 70000, 80000, 100000,
                                      160000,
                                      200000],
                         'max_iter': 240000}
        yolact_file = f'{self.out_dir}/yolact_config.yaml'
        with open(yolact_file, 'w') as yolact_fd:
            yaml.dump(yolact_config, yolact_fd)
        if validation_count > 0:
            validation_list = random.sample(self.image_files, validation_count)
            self.dumpCocoJson(validation_list, val_dir, ts)
        qw.QMessageBox.information(self, 'Data saved',
                                   f'Training images: {train_dir}\n'
                                   f'Validation images: {val_dir}\n'
                                   f'Yolact configuration: {yolact_file}')
        self.saved = True

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
            if findex not in self.seg_dict:
                continue
            img = cv2.imread(fpath)
            fname = os.path.basename(fpath)
            prefix = fname.rpartition('.')[0]
            # If image is bigger than allowed size, make some random crops
            xlist = [0]
            ylist = [0]
            if img.shape[0] > self.max_size:
                if img.shape[0] >= self.max_size * 1.5:
                    ylist = random.sample(range(img.shape[0] - self.max_size),
                                          4)
                else:
                    ylist = [img.shape[0] - self.max_size]
            if img.shape[1] > self.max_size:
                if img.shape[1] >= self.max_size * 1.5:
                    xlist = random.sample(range(img.shape[1] - self.max_size),
                                          4)
                else:
                    xlist = [img.shape[1] - self.max_size]
            for jj, (x, y) in enumerate(zip(xlist, ylist)):
                tmp_img = img[y: y + self.max_size, x: x + self.max_size]
                fname = f'{prefix}_{jj}.png'
                cv2.imwrite(os.path.join(imdir, fname),
                            tmp_img)
                coco['images'].append({
                    "license": 0,
                    "url": None,
                    "file_name": f"PNGImages/{fname}",
                    "height": img.shape[0],
                    "width": img.shape[1],
                    "date_captured": None,
                    "id": img_id
                })

                for seg in self.seg_dict[findex].values():
                    bbox = [int(xx) for xx in cv2.boundingRect(seg)]
                    if bbox[0] < y or bbox[0] + bbox[2] > y + self.max_size \
                            or bbox[1] < x or bbox[1] + bbox[
                        3] > x + self.max_size:
                        continue
                    bbox[0] -= y
                    bbox[1] -= x
                    tmp_seg = seg - [x, y]
                    annotation = {
                        "id": seg_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "segmentation": [[int(xx) for xx in tmp_seg.flatten()]],
                        "area": cv2.contourArea(tmp_seg),
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    coco['annotations'].append(annotation)
                    seg_id += 1
                img_id += 1
        with open(os.path.join(directory, 'annotations.json'), 'w') as fd:
            json.dump(coco, fd)


if __name__ == '__main__':
    app = qw.QApplication(sys.argv)
    win = TrainingWidget()
    win.setMinimumSize(800, 600)
    win.setWindowTitle('Argos - generate training data')
    win.showMaximized()
    app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())
