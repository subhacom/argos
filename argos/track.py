# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-05-28 3:01 PM
"""
======================
Track objects with GUI
======================

Usage:
::
    python -m argos.track


In Argos, this is the main tool for tracking objects
automatically. Argos tracks objects in two stages, first it segments
the individual objects (called instance segmentation) in a frame, and
then matches the positions of these segments to that in the previous
frame.

The segmentation can be done by a trained neural network via the
YOLACT library, or by classical image processing algorithms. Each of
these has its advantages and disadvantages.

Basic usage
-----------

This assumes you have a YOLACT network trained with images of your
target object. YOLACT comes with a network pretrained with a variety
of objects from the COCO database. If your target object is not
included in this, you can use the Argos annotation tool
(:py:mod:`argos.annotate`) to train a backbone network.

When you start Argos tracker, a window with an empty central widget is
presented (:numref:`track_startup`).

.. _track_startup:
.. figure:: ../doc/images/track_00.png
   :width: 100%
   :alt: Screenshot of tracking tool at startup

   Screenshot of tracking tool at startup


1. Use the ``File`` menu to open the desired video.  After selecting
   the video file, you will be prompted to:

    1. Select output data directory/file. You have a choice of CSV
       (text) or HDF5 (binary) format. HDF5 is recommended.

    2. Select Yolact configuration file, go to the `config` directory
       inside argos directory and select `yolact.yml`.

    3. File containing trained network weights, and here you should
       select the `babylocust_resnet101_119999_240000.pth` file.

2. This will show the first frame of the video in the central
   widget. On the right hand side you can set some parameters for the
   segmentation (:numref:`track_loaded`).

   .. _track_loaded:
   .. figure:: ../doc/images/track_01.png
      :width: 100%
      :alt: Tracking tool after loading video and YOLACT configuration and network weights.
   
      Tracking tool after loading video and YOLACT configuration and
      network weights.

   The top panel on the right is ``Yolact settings`` with the
   following fields:


   1. ``Number of objects to include``: keep at most these many
      detected objects.

   2. ``Detection score minimum``: YOLACT assigns a score between 0
      and 1 to each detected object to indicate how close it is to
      something the network is trained to detect. By setting this
      value higher, you can exclude spurious detection. Set it too
      high, and decent detections may be rejected.

   3. ``Merge overlaps more than``: If the bounding boxes of two
       detcted objects overlap more than this fraction of the smaller
       one, then consider them parts of the same object.

   The next panel, ``Size limits`` allows you to filter objects that
   are too big or too small. Here you can specify the minimum and
   maximum width and length of the bounding boxes, and any detection
   which does not fit will be removed.

   The bottom panel, ``SORTracker settings`` allows you to parametrize
   the actual tracking. SORTracker matches objects between frames by
   their distance. Default distance measure is ``Intersection over
   Union`` or IoU. This is the ratio of the area of intersection to
   the union of the two bounding boxes. 

   - ``Minimum overlap``: if the overlap between predicted position of
     an object and the actual detected position in the current frame is
     less than this, it is considered to be a new object. Thus, if an
     animal jumps from one position to a totally different position, the
     algorithm will think that a new object has appeared in the new
     location.

   - ``Minimum hits``: to avoid spurious detections, do not believe a
     detected object to be real unless it is detected in this many
     consecutive frames.

   - ``Maximum age``: if an object goes undetected for this many
     frames, remove it from the tracks, assuming it has gone out of
     view.


3. Start tracking: click the ``Play/Pause`` button and you should see
   the tracked objects with their bounding rectangles and Ids. The
   data will be saved in the filename you entered in step above
   (:numref:`track_running`).

   .. _track_running:
   .. figure:: ../doc/images/track_02.png
      :width: 100%
      :alt: Tracking in progress
   
      Tracking in progress. The bounding boxes of detected objects are
      outlined in green. Some spurious detections are visible which can
      be later corrected with the :py:mod:`argos.review` tool.


   If you choose CSV above, the bounding boxes of the segmented
   objects will be saved in ``{videofile}.seg.csv`` with each row
   containing `frame-no,x,y,w,h` where (x, y) is the coordinate of
   the top left corner of the bounding box and ``w`` and ``h`` are its
   width and height respectively.
   
   The tracks will be saved in ``{videofile}.trk.csv``. Each row in this
   file contains ``frame-no,track-id,x,y,w,h``.
   
   If you choose HDF5 instead, the same data will be saved in a single
   file compatible with the Pandas library. The segementation data
   will be saved in the group ``/segmented`` and tracks will be saved in
   the group ``/tracked``. The actual values are in the dataset named
   ``table`` inside each group, with columns in same order as described
   above for CSV file. You can load the tracks in a Pandas data frame
   in python with the code fragment:
   ::
           tracks = pandas.read_hdf(tracked_filename, 'tracked')


Classical segmentation
----------------------

Using the ``Segmentation method`` menu you can switch from YOLACT to
classical image segmentation for detecting target objects.  This
method uses patterns in the pixel values in the image to detect
contiguous patches. If your target objects are small but have high
contrast with the background, this may give tighter bounding boxes,
and thus more accurate tracking.
   
When this is enabled, the right panel will allow you to set the
parameters.  The parameters are detailed in
:py:mod:`argos.annotate`.

Briefly, the classical segmentation methods work by first converting
the image to gray-scale and then blurring the image so that sharp
edges of objects are smoothed out. The blurred image is then
thresholded using an adaptive method that adjusts the threshold value
based on local intensity. Thresholding produces a binary image which
is then processed to detect contiguous patches of pixels using one of
the available algorithms.

"""
import sys
import logging
import yaml
from PyQt5 import (
    QtWidgets as qw,
    QtCore as qc,
    QtGui as qg
)

import argos.utility as util
from argos.vwidget import VideoWidget
from argos.yolactwidget import YolactWidget
from argos.sortrackerwidget import SORTWidget
from argos.segwidget import SegWidget
from argos.csrtracker import CSRTWidget
from argos.limitswidget import LimitsWidget

# Set up logging for multithreading/multiprocessing
settings = util.init()


class ArgosMain(qw.QMainWindow):
    """"Main user interface"""

    sigQuit = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(ArgosMain, self).__init__(*args, **kwargs)
        self._video_widget = VideoWidget()
        self.setCentralWidget(self._video_widget)
        self._yolact_widget = YolactWidget()
        self._seg_widget = SegWidget()
        self._seg_widget.fixBboxOutline()

        self._lim_widget = LimitsWidget()

        self._sort_widget = SORTWidget()
        self._csrt_widget = CSRTWidget()

        self._yolact_dock = qw.QDockWidget('Yolact settings')
        self._yolact_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                          qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._yolact_dock)
        self._yolact_dock.setWidget(self._yolact_widget)

        self._seg_dock = qw.QDockWidget('Segmentation settings')
        self._seg_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                       qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._seg_dock)
        self._seg_scroll = qw.QScrollArea()
        self._seg_scroll.setWidgetResizable(True)
        self._seg_scroll.setWidget(self._seg_widget)
        self._seg_dock.setWidget(self._seg_scroll)

        self._lim_dock = qw.QDockWidget('Size limits')
        self._lim_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                       qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._lim_dock)
        self._lim_dock.setWidget(self._lim_widget)

        self._sort_dock = qw.QDockWidget('SORTracker settings')
        self._sort_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                        qc.Qt.RightDockWidgetArea)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._sort_dock)
        self._sort_scroll = qw.QScrollArea()
        self._sort_scroll.setWidgetResizable(True)
        self._sort_scroll.setWidget(self._sort_widget)
        self._sort_dock.setWidget(self._sort_scroll)

        self._csrt_dock = qw.QDockWidget('CSRTracker settings')
        self._csrt_dock.setAllowedAreas(qc.Qt.LeftDockWidgetArea |
                                        qc.Qt.RightDockWidgetArea)
        self._csrt_scroll = qw.QScrollArea()
        self._csrt_scroll.setWidgetResizable(True)
        self._csrt_scroll.setWidget(self._csrt_widget)
        self._csrt_dock.setWidget(self._csrt_scroll)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._csrt_dock)
        self._yolact_action = qw.QAction('Use Yolact segmentation')
        self._seg_action = qw.QAction('Use classical segmentation')
        self._seg_grp = qw.QActionGroup(self)
        self._seg_grp.addAction(self._yolact_action)
        self._seg_grp.addAction(self._seg_action)
        self._seg_grp.setExclusive(True)
        self._yolact_action.setCheckable(True)
        self._seg_action.setCheckable(True)
        self._yolact_action.setChecked(True)
        self._seg_dock.hide()
        self._sort_action = qw.QAction('Use SORT for tracking')
        self._sort_action.setCheckable(True)
        self._csrt_action = qw.QAction('Use CSRT for tracking')
        self._csrt_action.setCheckable(True)
        self._track_grp = qw.QActionGroup(self)
        self._track_grp.addAction(self._sort_action)
        self._track_grp.addAction(self._csrt_action)
        self._sort_action.setChecked(True)
        self._debug_action = qw.QAction('Debug')
        self._debug_action.setCheckable(True)
        debug_level = settings.value('track/debug', logging.INFO, type=int)
        self._debug_action.setChecked(debug_level == logging.DEBUG)
        logging.getLogger().setLevel(debug_level)
        self._debug_action.triggered.connect(self.setDebug)
        self._clear_settings_action = qw.QAction('Reset to default settings')
        self._clear_settings_action.triggered.connect(self.clearSettings)
        self._csrt_dock.hide()
        self._menubar = self.menuBar()
        self._file_menu = self._menubar.addMenu('&File')
        self._file_menu.addAction(self._video_widget.openAction)
        self._file_menu.addAction(self._video_widget.openCamAction)
        self.saveConfigAction = qw.QAction('Save configuration in file')
        self.saveConfigAction.triggered.connect(self.saveConfig)
        self._file_menu.addAction(self.saveConfigAction)
        self._seg_menu = self._menubar.addMenu('&Segmentation method')
        self._seg_menu.addActions(self._seg_grp.actions())
        self._track_menu = self._menubar.addMenu('&Tracking method')
        self._track_menu.addActions(self._track_grp.actions())
        self._view_menu = self._menubar.addMenu('View')
        self._view_menu.addActions([self._video_widget.zoomInAction,
                                    self._video_widget.zoomOutAction,
                                    self._video_widget.resetArenaAction,
                                    self._video_widget.showGrayscaleAction,
                                    self._video_widget.setColorAction,
                                    self._video_widget.autoColorAction,
                                    self._video_widget.colormapAction,
                                    self._video_widget.fontSizeAction,
                                    self._video_widget.relativeFontSizeAction,
                                    self._video_widget.lineWidthAction,
                                    self._video_widget.infoAction])
        self._advanced_menu = self.menuBar().addMenu('Advanced')
        # self._advanced_menu.addAction(self._video_widget.arenaSelectAction)
        self._advanced_menu.addAction(self._video_widget.resetArenaAction)
        self._advanced_menu.addAction(self._debug_action)
        self._advanced_menu.addAction(self._clear_settings_action)

        self.makeShortcuts()
        ##########################
        # Connections
        self._video_widget.sigVideoFile.connect(self.updateTitle)
        self._video_widget.sigSetFrame.connect(self._yolact_widget.process)
        self._video_widget.sigArena.connect(self._lim_widget.setRoi)
        self._video_widget.sigReset.connect(self._lim_widget.resetRoi)
        self._video_widget.sigArena.connect(self._seg_widget.setRoi)
        self._video_widget.sigReset.connect(self._seg_widget.resetRoi)
        self._video_widget.resetArenaAction.triggered.connect(
            self._seg_widget.resetRoi)
        self._yolact_widget.sigProcessed.connect(self._lim_widget.process)
        self._lim_widget.sigProcessed.connect(self._sort_widget.track)
        self._yolact_widget.sigProcessed.connect(
            self._video_widget.sigSetBboxes)
        self._lim_widget.sigWmin.connect(self._seg_widget.setWmin)
        self._lim_widget.sigWmax.connect(self._seg_widget.setWmax)
        self._lim_widget.sigHmin.connect(self._seg_widget.setHmin)
        self._lim_widget.sigHmax.connect(self._seg_widget.setHmax)
        self._seg_widget.sigProcessed.connect(self._sort_widget.sigTrack)
        self._seg_widget.sigProcessed.connect(self._video_widget.sigSetBboxes)
        # self._seg_widget.sigSegPolygons.connect(self._video_widget.sigSetSegmented)
        self._sort_widget.sigTracked.connect(self._video_widget.setTracked)
        self._csrt_widget.sigTracked.connect(self._video_widget.setTracked)
        self._video_widget.openAction.triggered.connect(
            self._sort_widget.sigReset)
        self._video_widget.openAction.triggered.connect(
            self._csrt_widget.sigReset)
        self._video_widget.sigReset.connect(self._sort_widget.sigReset)
        self._video_widget.sigReset.connect(self._csrt_widget.sigReset)
        self._video_widget.sigStatusMsg.connect(self.statusMsgSlot)
        self._seg_grp.triggered.connect(self.switchSegmentation)
        self._track_grp.triggered.connect(self.switchTracking)
        self.sigQuit.connect(self._video_widget.sigQuit)
        self.sigQuit.connect(self._yolact_widget.sigQuit)
        self.sigQuit.connect(self._seg_widget.sigQuit)
        self.sigQuit.connect(self._lim_widget.sigQuit)
        self.sigQuit.connect(self._sort_widget.sigQuit)
        self.sigQuit.connect(self._csrt_widget.sigQuit)

    def makeShortcuts(self):
        self.sc_play = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Space), self)
        self.sc_play.activated.connect(self._video_widget.playAction.trigger)
        self.sc_open = qw.QShortcut(qg.QKeySequence('Ctrl+O'), self)
        self.sc_open.activated.connect(self._video_widget.openVideo)

    @qc.pyqtSlot(bool)
    def setDebug(self, state):
        level = logging.DEBUG if state else logging.INFO
        settings.setValue('track/debug', level)
        logging.getLogger().setLevel(level)

    def cleanup(self):
        self.sigQuit.emit()
        settings.sync()
        logging.debug('Saved settings')

    @qc.pyqtSlot()
    def clearSettings(self):
        button = qw.QMessageBox.question(
            self,
            'Reset settings to default',
            'Are you sure to clear all saved settings?')
        if button == qw.QMessageBox.NoButton:
            return
        settings.clear()
        settings.sync()

    @qc.pyqtSlot(qw.QAction)
    def switchSegmentation(self, action):
        """Switch segmentation widget between yolact and classical"""
        self._video_widget.pauseVideo()
        if action == self._yolact_action:
            util.reconnect(self._video_widget.sigSetFrame,
                           newhandler=self._yolact_widget.process,
                           oldhandler=self._seg_widget.sigProcess)
            self._yolact_dock.setVisible(True)
            self._seg_dock.setVisible(False)
        else:  # classical segmentation, self._seg_action
            util.reconnect(self._video_widget.sigSetFrame,
                           newhandler=self._seg_widget.sigProcess,
                           oldhandler=self._yolact_widget.process)
            self._yolact_dock.setVisible(False)
            self._seg_dock.setVisible(True)

    @qc.pyqtSlot(qw.QAction)
    def switchTracking(self, action):
        """Switch tracking between SORT and CSRT"""
        self._video_widget.pauseVideo()
        if action == self._sort_action:
            self._sort_dock.show()
            self._csrt_dock.hide()
            try:
                self._video_widget.sigSetFrame.disconnect(
                    self._csrt_widget.setFrame)
            except TypeError:
                pass
            newhandler = self._sort_widget.sigTrack
            oldhandler = self._csrt_widget.setBboxes
        else:  # csrt
            self._csrt_dock.show()
            self._sort_dock.hide()
            self._video_widget.sigSetFrame.connect(self._csrt_widget.setFrame)
            newhandler = self._csrt_widget.setBboxes
            oldhandler = self._sort_widget.sigTrack
        if self._yolact_action.isChecked():
            sig = self._lim_widget.sigProcessed
        else:
            sig = self._seg_widget.sigProcessed
        try:
            sig.disconnect()
        except TypeError:
            pass
        util.reconnect(sig, newhandler, oldhandler)

    @qc.pyqtSlot(str)
    def updateTitle(self, filename: str) -> None:
        self.setWindowTitle(f'Argos:track {filename}')

    @qc.pyqtSlot()
    def saveConfig(self):
        config = {
            'wmin': settings.value('segment/min_width', 10, type=int),
            'wmax': settings.value('segment/max_width', 50, type=int),
            'hmin': settings.value('segment/min_height', 10, type=int),
            'hmax': settings.value('segment/max_height', 100, type=int),
            'sort_metric': settings.value('sortracker/metric', 'iou', type=str),
            'min_dist': settings.value('sortracker/min_dist', 0.3, type=float),
            'min_hits': settings.value('sortracker/min_hits', 3, type=int),
            'max_age': settings.value('sortracker/max_age', 10, type=int),
            'pmin': settings.value('segment/min_pixels', 10, type=int),
            'pmax': settings.value('segment/max_pixels', 1000, type=int)
        }
        if self._seg_dock.isVisible():
            config['blur_width'] = settings.value('segment/blur_width', 21,
                                                  type=int)
            config['blur_sd'] = settings.value('segment/blur_sd', 1.0,
                                               type=float)
            config['thresh_method'] = settings.value('segment/thresh_method',
                                                     'gaussian', type=str)
            config['thresh_max'] = settings.value(
                'segment/thresh_max_intensity', 255, type=int)
            config['thresh_baseline'] = settings.value(
                'segment/thresh_baseline', type=int)
            config['thresh_blocksize'] = settings.value(
                'segment/thresh_blocksize', type=int)
            config['thresh_invert'] = settings.value('segment/thresh_invert',
                                                     True,
                                                     type=bool)
            seg_method = settings.value('segment/method', 'threshold',
                                        type=str).lower()
            config['seg_method'] = seg_method
            if seg_method == 'watershed':
                config['dist_thresh'] = int(settings.value(
                    'segment/watershed_distthresh', 3, type=float))
            elif seg_method == 'dbscan':
                config['eps'] = settings.value('segment/dbscan_eps', 5.0,
                                               type=float)
                config['min_samples'] = settings.value(
                    'segment/dbscan_minsamples', 10, type=int)
        else:  # yolact
            config['method'] = 'yolact'
            config['weight'] = settings.value('yolact/weightsfile', type=str)
            config['yconfig'] = settings.value('yolact/configfile', type=str)
            config['overlap_thresh'] = settings.value('yolact/overlap_thresh',
                                                      1.0, type=float)
            config['score_thresh'] = settings.value('yolact/score_thresh', 0.1,
                                                    type=float)
            config['top_k'] = settings.value('yolact/top_k', 30,
                                             type=int)
            config['cuda'] = settings.value('yolact/cuda', True, type=bool)

        directory = settings.value('video/directory', '.')
        fname, _ = qw.QFileDialog.getSaveFileName(
            self, 'Save configuration as',
            directory,
            filter='yaml (*.yml *.yaml)')
        if len(fname) > 0:
            with open(fname, 'w') as fd:
                yaml.dump(config, fd)

    @qc.pyqtSlot()
    def loadConfig(self):
        """TODO implement loading config file exported in yaml file"""
        raise NotImplementedError('This function is yet to be implemented')

    @qc.pyqtSlot(str)
    def statusMsgSlot(self, msg):
        self.statusBar().showMessage(msg)


if __name__ == '__main__':
    app = qw.QApplication(sys.argv)
    win = ArgosMain()
    win.setMinimumSize(800, 600)
    win.setWindowTitle('Argos - track animals in video')
    win.showMaximized()
    app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())
