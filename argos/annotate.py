# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-29 4:52 PM

"""
=================================
Generate training data for YOLACT
=================================
Usage:
::
    python -m argos.annotate

This program helps you annotate a set of images and export the images and
annotation in a way that YOLACT can process for training. Note that this is for
a single category of objects.

Preparation
===========
Create a folder and copy all the images you want to annotate into it.

If you have videos instead, you need to extract some video
frames. There are many programs, including most video players, which
can do this. Argos includes a small utility script
:py:mod:`argos.extract_frames` if you need.

Upon startup the program will prompt you to choose the folder
containing the images to be annotated. Browse to the desired image
folder. All the images should be directly in this folder, no
subfolders.

If you want to extract training images from a recorded video file, choose
``Cancel`` in this dialog and select ``File->Extract from video`` in the menu.
This will ask you to choose the video file, then the number of frames you want
to extract, whether the frames should be randomly selected or not, and the
folder to dump them in. Once finished, the output folder is set as the input
image folder.

Annotate new images
===================
After you select the image folder, the annotator will show you the
main window, with an empty display like below.

.. figure:: ../doc/images/annotate_00.png
   :width: 100%
   :alt: Screenshot of annotate tool at startup

   Screenshot of annotate tool at startup

The ``Files/Dirs`` pane on the bottom right lists all the files in the
image directory selected at startup. (Note that this pane may take up
too much screen space. You can close any of the panes using the 'x'
button on their titlebar or or move them around by dragging them with
left mouse button).

The ``Segmentation settings`` pane on right allows you to choose the
parameters for segmentation. See below for details on these settings.


You can press ``PgDn`` key, or click on any of the file names listed
in ``Files/Dirs`` pane to start segmenting the image files. Keep
pressing ``PgDn`` to go to next image, and ``PgUp`` to go back to
previous image.

It can take about a second to segment an image, depending on the image
size and the method of segmentation. Once the image is segmented, the
segment IDs will be listed in ``Segmented objects`` pane on the left.

.. figure:: ../doc/images/annotate_01.png
   :width: 100%
   :alt: Screenshot of annotate tool after segmenting an image.

   Screenshot of annotate tool after segmenting an image.

The image above shows some locusts in a box with petri dishes
containing paper strips. As you can see, the segmentation includes
spots on the paper floor, edges of the petri dishes as well as the
animals. 

We want to train the YOLACT network to detect the locusts. So we must
remove any segmented objects that are not locusts. To do this, click on
the ID of an unwanted object on the left pane listing ``Segmented
objects``. The selected object will be outlined with dotted blue line. 

You can click the ``Remove selected objects`` button on the panel at
the bottom left, or press ``x`` on the keyboard to delete this
segmented object.

.. figure:: ../doc/images/annotate_02.png
   :width: 100%
   :alt: Screenshot of annotate tool for selecting a segmented object.

   Screenshot of annotate tool for selecting a segmented
   object. Segmented object 16 is part of the petri-dish edge and we
   want to exclude it from the list of annotated objects in this
   image.

Alternatively, if the number of animals is small compared to the
spuriously segmented objects, you can select all the animals by
keeping the ``Ctrl`` key pressed while left-clicking on the IDs of the
animals on the left pane. Then click ``Keep selected objects`` or
press ``k`` on the keyboard to delete all other segmented
objects.

By default, objects are outlined with solid green line, and selected
objects are outlined with dotted blue line. But you can change this
from ``View`` menu. 

In the ``View`` menu you can check ``Autocolor`` to make the program
automatically use a different color for each object. In this case, the
selected object is outlined in a thicker line of the same color, while
all other object outlines are dimmed.

You can also choose ``Colormap`` from the view menu and specify the
number of colors to use. Each object will be outlined in one of these
colors, going back to the first color when all the colors have been
used.

Segmentation settings
---------------------

The segmentation settings pane allows you to control how each image is
segmented. The segmentation here is done in the following steps:

1. Convert the image to gray-scale 

2. Smooth the gray-scale image by Gaussian blurring. For this the
   following parameters can be set:

   - Blur width: diameter of the 2D Gaussian window in pixels 

   - Blur sd: Standard deviation of the Gaussian curve used for
     blurring.

3. Threshold the blurred image. For this the following parameters can
   be set:

   - Invert thresholding: instead of taking the pixels above threshold
     value, take those below. This should be checked when the objects
     of interest are darker than background.

   - Thresholding method: Choice between Adaptive Gaussian and
     Adaptive Mean. These are the two adaptive thresholding methods
     provided by the OpenCV library. In practice it does not seem to
     matter much.

   - Threshold maximum intensity: pixel values above threshold are set
     to this value. It matters only for the Watershed algorithm for
     segmentation (see below). Otherwise, any value above the threshold
     baseline is fine.

   - Threshold baseline: the actual threshold value for each pixel is
     based on this value. When using adaptive mean, the threshold
     value for a pixel is the mean value in its ``block size``
     neighborhood minus this baseline value. For adaptive Gaussian,
     the threshold value is the Gaussian-weighted sum of the values in
     its neighborhood minus this baseline value.

   - Thresholding block size: size of the neighborhood considered for
     each pixel.

   - Segmentation method: This combo box allows you to choose between
     several thresholding methods. 

     - ``Threshold`` and ``Contour`` are essentially the same, with
       slight difference in speed. They both find the blobs in the
       thresholded image and consider them as objects.

     - ``Watershed`` uses the watershed algorithm from OpenCV
       library. This is good for objects covering large patches (100s
       of pixels) in the image, but not so good for very small
       objects. It is also slower than ``Contour/Thresholding``
       methods.

     - ``DBSCAN`` uses the DBSCAN clustering algorithm from
       ``scikit-learn`` package to spatially cluster the non-zero
       pixels in the thresholded image. This is the slowest method,
       but may be good for intricate structures (for example legs of
       insects in an image are often missed by the other algorithms,
       but DBSCAN may keep them depending on the parameter
       settings). When you choose this method, there are additional
       parameters to be specified. For a better understanding of
       DBSCAN algorithm and relevant references see its documentation
       in ``scikit-learn`` package:
       https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
       
       - DBSCAN minimum samples: The core points of a cluster should
         include these many neighbors.

       - DBSCAN epsilon: this is the neighborhood size, i.e., each
         core point of a cluster should have ``minimum samples``
         neighbors within this radius. Experiment with it (try values
         like 0.1, 1, 5, etc)!

    - Minimum pixels: filter out segmented objects with fewer than
      these many pixels.

    - Maximum pixels: filter out segmented objects with more than
      these many pixels.

    - Show intermediate steps: used for debugging. Default is
      ``Final`` which does nothing. Other choices, ``Blurred``,
      ``Thresholded``, ``Segmented`` and ``Filtered`` show the output
      of the selected step in a separate window.

    - Boundary style: how to show the boundary of the objects. Default
      is ``contour``, which outlines the segmented objects. ``bbox``
      will show the bounding horizontal rectangles, ``minrect`` will
      show smallest rectangles bounding the objects at any angle, and
      ``fill`` will fill the contours of the objects with color.


    - Minimum width: the smaller side of the bounding rectangle of an
      object should be greater or equal to these many pixels.

    - Maximum width: the smaller side of the bounding rectangle of an
      object should be less than these many pixels.

    - Minimum length: the bigger side of the bounding rectangle of an
      object should be greater or equal to these many pixels.

    - Maximum length: the bigger side of the bounding rectangle of an
      object should be less than these many pixels.

Save segmentation
-----------------

You can save all the data for currently segmented images in a file by
pressing ``Ctrl+S`` on keyboard or selecting ``File->Save segmentation`` from the
menu bar. This will be a Python pickle file (extension ``.pkl`` or
``.pickle``).

Load segmentation
-----------------

You can load segmentation data saved before by pressing ``Ctrl+O`` on
keyboard or by selecting ``File->Open saved segmentation`` from the
menu bar.

Export training and validation data
-----------------------------------

Press ``Ctrl+E`` on keyboard or select ``File->Export training and
validation data`` from menubar to export the annotation data in a
format that YOLACT can read for training.

This will prompt you to choose an export directory. Once that is done,
it will bring up a dialog box as below for you to enter some metadata
and the split of training and validation set.

.. figure:: ../doc/images/annotate_03.png
   :width: 100%
   :alt: Screenshot of export dialog

   Screenshot of annotate tool export annotation dialog


- ``Object class``: here, type in the name of the objects of interest.

- ``Neural-Net base configuration``: select the backbone neural
  network if you are trying something new. The default
  ``yolact_base_config`` should work with the pretrained ``resnet
  101`` based network that is distributd with YOLACT. Other options
  have not been tested much.

- ``Use % of images for validation``: by default we do a 70-30 split
  of the available images. That is 70% of the images are used for
  training and 30% for validation.

- ``Split into subregions``: when the image is bigger than the neural
  network's input size (550x550 pixels in most cases), randomly split
  the image into blocks of this size, taking care to keep at least one
  segmented object in each block. These blocks are then saved as
  individual training images.

- ``Export boundaries as``: you can choose to give the detailed
  contour of each segmented object, or its axis-aligned bounding
  rectangle, or its minimum-area rotated bounding rectangle
  here. Contour provides the most information.

  Once done, you will see a message titled ``Data saved`` showing the
  command to be used for training YOLACT. It is also copied to the
  clipboard, so you can just use the ``paste`` action on your
  operating system to run the training from a command line.

.. figure:: ../doc/images/annotate_04.png
   :width: 100%
   :alt: Screenshot of suggested command line after exporting annotations.

   Screenshot of suggested command line after exporting annotations.

"""
import sys
# import time
import logging
import os
# from collections import OrderedDict
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

from argos.constants import (
    OutlineStyle,
    DrawingGeom)

from argos import utility as ut
from argos import constants as const
# from argos import frameview
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
    sigSetDisplayGeom = qc.pyqtSignal(DrawingGeom)

    def __init__(self, *args, **kwargs):
        super(TrainingWidget, self).__init__(*args, **kwargs)
        self._waiting = False
        self.boundaryType = 'contour'
        self.displayCoco = True
        self.numCrops = 1  # number of random crops to generate if input image is bigger than training image size
        self.saved = True
        self.validationFrac = 0.3
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
        self.validationDir = 'validation'
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
        self.segWidget.setSegmentationMethod('Contour')
        self.segWidget.outlineCombo.setCurrentText('contour')
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
        self.displayWidget.frameScene.linewidth = 0
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
        self._makeConnections()
        self.openImageDir()
        self.statusBar().showMessage('Press `Next image` to start segmenting')

    def _makeFileDock(self):
        self.fileDock = qw.QDockWidget('Files/Dirs')

        dirlayout = qw.QFormLayout()
        self.outDirLabel = qw.QLabel('Output directory for training data')
        self.outDirName = qw.QLabel(self.outputDir)
        self.outDirName.setMinimumWidth(0)
        self.outDirName.setSizePolicy(qw.QSizePolicy.MinimumExpanding,
                                      qw.QSizePolicy.Preferred)
        dirlayout.addRow(self.outDirLabel, self.outDirName)
        self.imageDirLabel = qw.QLabel('Input image directory')
        self.imageDirName = qw.QLabel(self.imageDir)
        self.imageDirName.setMinimumWidth(0)
        self.imageDirName.setSizePolicy(qw.QSizePolicy.MinimumExpanding,
                                        qw.QSizePolicy.Preferred)
        dirlayout.addRow(self.imageDirLabel, self.imageDirName)
        self.dirWidget = qw.QWidget()
        self.dirWidget.setLayout(dirlayout)

        self.fileView = qw.QListView()
        self.fileView.setSizeAdjustPolicy(qw.QListWidget.AdjustToContents)
        self.fileModel = qw.QFileSystemModel()
        self.fileModel.setFilter(qc.QDir.NoDotAndDotDot | qc.QDir.Files)
        self.fileView.setModel(self.fileModel)
        self.fileView.setRootIndex(self.fileModel.setRootPath(self.imageDir))
        self.fileView.selectionModel().selectionChanged.connect(self.handleFileSelectionChanged)

        self.fwidget = qw.QWidget()
        layout = qw.QVBoxLayout()
        layout.addWidget(self.dirWidget)
        layout.addWidget(self.fileView)
        self.fwidget.setLayout(layout)
        self.fileDock.setWidget(self.fwidget)
        self.fileDock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                      qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self.fileDock)

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
        self.clearCurButton = qw.QToolButton()
        self.clearCurButton.setSizePolicy(qw.QSizePolicy.Minimum,
                                          qw.QSizePolicy.MinimumExpanding)
        self.clearCurButton.setDefaultAction(self.clearCurrentAction)
        self.clearAllButton = qw.QToolButton()
        self.clearAllButton.setSizePolicy(qw.QSizePolicy.Minimum,
                                          qw.QSizePolicy.MinimumExpanding)
        self.clearAllButton.setDefaultAction(self.clearAllAction)
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
        layout.addWidget(self.clearCurButton)
        layout.addWidget(self.clearAllButton)
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
        self.extractFramesAction = qw.QAction('Extract frames from video')
        self.extractFramesAction.triggered.connect(self.extractFrames)
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

        # Views of the dock widgets
        self.showFileDockAction = qw.QAction('Directory listing')
        self.showFileDockAction.setCheckable(True)
        self.showFileDockAction.setChecked(True)
        self.showSegDockAction = qw.QAction('Segmentation settings')
        self.showSegDockAction.setCheckable(True)
        self.showSegDockAction.setChecked(True)
        self.showActionsDockAction = qw.QAction('Segmentation actions')
        self.showActionsDockAction.setCheckable(True)
        self.showActionsDockAction.setChecked(True)

        self.showIntermediateAction = qw.QAction('Show intermediate result')
        self.showIntermediateAction.setCheckable(True)
        self.showIntermediateAction.setChecked(False)
        self.showIntermediateAction.triggered.connect(
            self.segWidget.showIntermediateOutput)

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
        self.resegmentCurrentImageKey = qw.QShortcut(qg.QKeySequence('R'), self)
        self.resegmentCurrentImageKey.activated.connect(self.resegmentCurrent)

        self.saveKey = qw.QShortcut(qg.QKeySequence('Ctrl+S'), self)
        self.saveKey.activated.connect(
            self.saveSegmentation)
        self.openKey = qw.QShortcut(qg.QKeySequence('Ctrl+O'), self)
        self.openKey.activated.connect(
            self.loadSegmentation)

        self.exportKey = qw.QShortcut(qg.QKeySequence('Ctrl+E'), self)
        self.exportKey.activated.connect(self.exportSegmentation)

    def _makeConnections(self):
        """separated from creating actions because they refer to objects that
        are created after the actions."""
        self.showFileDockAction.triggered.connect(self.fileDock.setVisible)
        self.showSegDockAction.triggered.connect(self.segDock.setVisible)
        self.showActionsDockAction.triggered.connect(self.fileDock.setVisible)

    def _makeMenuBar(self):
        self.fileMenu = self.menuBar().addMenu('&File')
        self.fileMenu.addActions([self.imagedirAction,
                                  self.outdirAction,
                                  self.extractFramesAction,
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
                                  self.displayWidget.setColorAction,
                                  self.displayWidget.setSelectedColorAction,
                                  self.displayWidget.setAlphaUnselectedAction,
                                  self.displayWidget.autoColorAction,
                                  self.displayWidget.colormapAction,
                                  self.displayWidget.lineWidthAction,
                                  self.displayWidget.fontSizeAction,
                                  self.displayWidget.relativeFontSizeAction,
                                  self.displayWidget.setLabelInsideAction,
                                  self.showIntermediateAction])
        self.docksMenu = self.viewMenu.addMenu('Dock widgets')
        self.docksMenu.addActions([self.showFileDockAction,
                                  self.showSegDockAction,
                                  self.showActionsDockAction])
        self.advancedMenu = self.menuBar().addMenu('Advanced')
        self.advancedMenu.addAction(self.debugAction)

    @qc.pyqtSlot(bool)
    def setDebug(self, val: bool):
        level = logging.DEBUG if val else logging.INFO
        logging.getLogger().setLevel(level)
        settings.setValue('ytrainer/debug', level)

    def outlineStyleToBoundaryMode(self, style):
        if style == OutlineStyle.bbox:
            self.sigSetDisplayGeom.emit(DrawingGeom.rectangle)
        else:
            self.sigSetDisplayGeom.emit(DrawingGeom.polygon)

    def _openImageDir(self, directory):
        logging.debug(f'Opening directory "{directory}"')
        try:
            self.imageFiles = [entry.path for entry in
                               os.scandir(directory)]
            self.imageIndex = -1
            settings.setValue('training/imagedir', directory)
            self.imageDir = directory
            self.imageDirName.setText(directory)
        except IOError as err:
            qw.QMessageBox.critical(self, 'Could not open image directory',
                                    str(err))
        self.fileView.setRootIndex(self.fileModel.setRootPath(self.imageDir))

    def openImageDir(self):
        directory = settings.value('training/imagedir', '.')
        tmp = qw.QFileDialog.getExistingDirectory(self,
                                                        'Open image diretory',
                                                        directory=directory)
        if len(tmp) > 0:
            directory = tmp
        self._openImageDir(directory)

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
            self.outDirName.setText(directory)
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
        fname = self.fileModel.data(indices[0])
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
        self.segWidget.hideIntermediateOutput()
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
        valText = qw.QLineEdit(str(int(self.validationFrac * 100)))

        def setValFrac():
            self.validationFrac = float(valText.text()) / 100

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
        validationCount = int(len(self.imageFiles) * self.validationFrac)
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

    @qc.pyqtSlot()
    def extractFrames(self):
        """Extract frames from a video and open the extraction directory for
        annotating images"""
        saveDir = settings.value('training/savedir', '.')

        filename, _ = qw.QFileDialog.getOpenFileName(
                self, 'Load video', directory=saveDir,
                filter='Video file (*.avi *.mp4 *.mpg *.mpeg *.ogg *.webm *.wmv'
                       ' *.mov);;All files (*)')
        if len(filename) == 0:
            return
        nframes, ok = qw.QInputDialog.getInt(self, 'Extract frames',
                                         'Number of frames to extract',
                                         const.EXTRACT_FRAMES, min=0)
        if not ok:
            return
        scale, _ =  qw.QInputDialog.getDouble(self, 'Extract frames',
                                         'Scale each frame X',
                                         1.0, min=0.1)
        if not ok:
            return
        random = qw.QMessageBox.question(self, 'Random frames?',
                                         'Select random frames?',
                                         qw.QMessageBox.Yes | qw.QMessageBox.No)
        extractDir = qw.QFileDialog.getExistingDirectory(self,
            'Extract frames into directory',
            saveDir
        )
        if len(extractDir) == 0:
            return
        ut.extract_frames(filename, nframes, scale, extractDir,
                          random=qw.QMessageBox.Yes)
        self._openImageDir(extractDir)



if __name__ == '__main__':
    app = qw.QApplication(sys.argv)
    win = TrainingWidget()
    win.setMinimumSize(800, 600)
    win.setWindowTitle('Argos - generate training data')
    win.showMaximized()
    app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())
