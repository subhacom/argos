# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-05-28 3:01 PM

import sys
import logging
import yaml
from PyQt5 import QtWidgets as qw, QtCore as qc, QtGui as qg

import argos.utility as util
from argos.vwidget import VideoWidget
from argos.yolactwidget import YolactWidget
from argos.yolov11widget import Yolov11Widget
from argos.sortrackerwidget import SORTWidget
from argos.bytetrackerwidget import ByteTrackerWidget
from argos.ocsortwidget import OCSORTWidget
from argos.segwidget import SegWidget
from argos.csrtracker import CSRTWidget
from argos.limitswidget import LimitsWidget

# Set up logging for multithreading/multiprocessing
settings = util.init()


class ArgosTracker(qw.QMainWindow):
    """ "Main user interface"""

    sigQuit = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(ArgosTracker, self).__init__(*args, **kwargs)
        self._video_widget = VideoWidget()
        self.setCentralWidget(self._video_widget)
        self._outfile_label = qw.QLabel('')
        self._yolov11_widget = Yolov11Widget()
        self._yolact_widget = YolactWidget()
        self._seg_widget = SegWidget()
        self._seg_widget.fixBboxOutline()

        self._lim_widget = LimitsWidget()

        self._sort_widget = SORTWidget()
        self._bytetrack_widget = ByteTrackerWidget()
        self._ocsort_widget = OCSORTWidget()
        self._csrt_widget = CSRTWidget()

        self._yolov11_dock = qw.QDockWidget('YOLOv11 settings')
        self._yolov11_dock.setAllowedAreas(
            qc.Qt.LeftDockWidgetArea | qc.Qt.RightDockWidgetArea
        )
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._yolov11_dock)
        self._yolov11_dock.setWidget(self._yolov11_widget)

        self._yolact_dock = qw.QDockWidget('Yolact settings')
        self._yolact_dock.setAllowedAreas(
            qc.Qt.LeftDockWidgetArea | qc.Qt.RightDockWidgetArea
        )
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._yolact_dock)
        self._yolact_dock.setWidget(self._yolact_widget)

        self._seg_dock = qw.QDockWidget('Segmentation settings')
        self._seg_dock.setAllowedAreas(
            qc.Qt.LeftDockWidgetArea | qc.Qt.RightDockWidgetArea
        )
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._seg_dock)
        self._seg_scroll = qw.QScrollArea()
        self._seg_scroll.setWidgetResizable(True)
        self._seg_scroll.setWidget(self._seg_widget)
        self._seg_dock.setWidget(self._seg_scroll)

        self._lim_dock = qw.QDockWidget('Size limits')
        self._lim_dock.setAllowedAreas(
            qc.Qt.LeftDockWidgetArea | qc.Qt.RightDockWidgetArea
        )
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._lim_dock)
        self._lim_dock.setWidget(self._lim_widget)

        self._sort_dock = qw.QDockWidget('SORTracker settings')
        self._sort_dock.setAllowedAreas(
            qc.Qt.LeftDockWidgetArea | qc.Qt.RightDockWidgetArea
        )
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._sort_dock)
        self._sort_scroll = qw.QScrollArea()
        self._sort_scroll.setWidgetResizable(True)
        self._sort_scroll.setWidget(self._sort_widget)
        self._sort_dock.setWidget(self._sort_scroll)

        self._bytetrack_dock = qw.QDockWidget('ByteTracker settings')
        self._bytetrack_dock.setAllowedAreas(
            qc.Qt.LeftDockWidgetArea | qc.Qt.RightDockWidgetArea
        )
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._bytetrack_dock)
        self._bytetrack_scroll = qw.QScrollArea()
        self._bytetrack_scroll.setWidgetResizable(True)
        self._bytetrack_scroll.setWidget(self._bytetrack_widget)
        self._bytetrack_dock.setWidget(self._bytetrack_scroll)

        self._ocsort_dock = qw.QDockWidget('OC-SORT settings')
        self._ocsort_dock.setAllowedAreas(
            qc.Qt.LeftDockWidgetArea | qc.Qt.RightDockWidgetArea
        )
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._ocsort_dock)
        self._ocsort_scroll = qw.QScrollArea()
        self._ocsort_scroll.setWidgetResizable(True)
        self._ocsort_scroll.setWidget(self._ocsort_widget)
        self._ocsort_dock.setWidget(self._ocsort_scroll)

        self._csrt_dock = qw.QDockWidget('CSRTracker settings')
        self._csrt_dock.setAllowedAreas(
            qc.Qt.LeftDockWidgetArea | qc.Qt.RightDockWidgetArea
        )
        self._csrt_scroll = qw.QScrollArea()
        self._csrt_scroll.setWidgetResizable(True)
        self._csrt_scroll.setWidget(self._csrt_widget)
        self._csrt_dock.setWidget(self._csrt_scroll)
        self.addDockWidget(qc.Qt.RightDockWidgetArea, self._csrt_dock)
        self._yolov11_action = qw.QAction('Use YOLOv11 segmentation')
        self._yolact_action = qw.QAction('Use Yolact segmentation')
        self._seg_action = qw.QAction('Use classical segmentation')
        self._seg_grp = qw.QActionGroup(self)
        self._seg_grp.addAction(self._yolov11_action)
        self._seg_grp.addAction(self._yolact_action)
        self._seg_grp.addAction(self._seg_action)
        self._seg_grp.setExclusive(True)
        self._yolov11_action.setCheckable(True)
        self._yolact_action.setCheckable(True)
        self._seg_action.setCheckable(True)
        self._yolov11_action.setChecked(True)
        self._yolact_dock.hide()
        self._seg_dock.hide()
        self._bytetrack_action = qw.QAction('Use ByteTrack for tracking')
        self._bytetrack_action.setCheckable(True)
        self._ocsort_action = qw.QAction('Use OC-SORT for tracking')
        self._ocsort_action.setCheckable(True)
        self._sort_action = qw.QAction('Use SORT for tracking')
        self._sort_action.setCheckable(True)
        self._csrt_action = qw.QAction('Use CSRT for tracking')
        self._csrt_action.setCheckable(True)
        self._track_grp = qw.QActionGroup(self)
        self._track_grp.addAction(self._bytetrack_action)
        self._track_grp.addAction(self._ocsort_action)
        self._track_grp.addAction(self._sort_action)
        self._track_grp.addAction(self._csrt_action)
        self._bytetrack_action.setChecked(True)
        self._sort_dock.hide()
        self._ocsort_dock.hide()
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
        self._quit_action = qw.QAction('Quit')
        self._file_menu.addAction(self._quit_action)
        self._seg_menu = self._menubar.addMenu('&Segmentation method')
        self._seg_menu.addActions(self._seg_grp.actions())
        self._seg_menu.setToolTipsVisible(True)
        self._track_menu = self._menubar.addMenu('&Tracking method')
        self._track_menu.addActions(self._track_grp.actions())
        self._view_menu = self._menubar.addMenu('View')
        self._view_menu.addActions(
            [
                self._video_widget.zoomInAction,
                self._video_widget.zoomOutAction,
                self._video_widget.resetArenaAction,
                self._video_widget.showGrayscaleAction,
                self._video_widget.setColorAction,
                self._video_widget.autoColorAction,
                self._video_widget.colormapAction,
                self._video_widget.setLabelInsideAction,
                self._video_widget.fontSizeAction,
                self._video_widget.relativeFontSizeAction,
                self._video_widget.lineWidthAction,
                self._video_widget.infoAction,
            ]
        )
        self._play_menu = self.menuBar().addMenu('&Play')
        self._play_menu.addActions(
            (self._video_widget.playAction, self._video_widget.refreshAction)
        )
        self._advanced_menu = self.menuBar().addMenu('Advanced')
        # self._advanced_menu.addAction(self._video_widget.arenaSelectAction)
        self._advanced_menu.addAction(self._video_widget.resetArenaAction)
        self._advanced_menu.addAction(self._debug_action)
        self._advanced_menu.addAction(self._clear_settings_action)
        self.statusBar().addPermanentWidget(self._outfile_label)

        self.makeShortcuts()
        ##########################
        # Connections
        self._video_widget.sigVideoFile.connect(self.updateTitle)
        self._video_widget.sigSetFrame.connect(self._yolov11_widget.process)
        self._video_widget.sigArena.connect(self._lim_widget.setRoi)
        self._video_widget.sigReset.connect(self._lim_widget.resetRoi)
        self._video_widget.sigArena.connect(self._seg_widget.setRoi)
        self._video_widget.sigReset.connect(self._seg_widget.resetRoi)
        self._video_widget.resetArenaAction.triggered.connect(
            self._seg_widget.resetRoi
        )
        # Downstream connections are permanent; only the active segmentation
        # widget receives frames (via sigSetFrame) so only it emits sigProcessed.
        self._yolov11_widget.sigProcessed.connect(self._lim_widget.process)
        self._yolov11_widget.sigProcessed.connect(self._video_widget.sigSetBboxes)
        self._yolact_widget.sigProcessed.connect(self._lim_widget.process)
        self._lim_widget.sigProcessed.connect(self._bytetrack_widget.track)
        self._yolact_widget.sigProcessed.connect(
            self._video_widget.sigSetBboxes
        )
        self._lim_widget.sigWmin.connect(self._seg_widget.setWmin)
        self._lim_widget.sigWmax.connect(self._seg_widget.setWmax)
        self._lim_widget.sigHmin.connect(self._seg_widget.setHmin)
        self._lim_widget.sigHmax.connect(self._seg_widget.setHmax)
        self._seg_widget.sigProcessed.connect(self._bytetrack_widget.sigTrack)
        self._seg_widget.sigProcessed.connect(self._video_widget.sigSetBboxes)
        # self._seg_widget.sigSegPolygons.connect(self._video_widget.sigSetSegmented)
        self._bytetrack_widget.sigTracked.connect(self._video_widget.setTracked)
        self._ocsort_widget.sigTracked.connect(self._video_widget.setTracked)
        self._sort_widget.sigTracked.connect(self._video_widget.setTracked)
        self._csrt_widget.sigTracked.connect(self._video_widget.setTracked)
        self._video_widget.openAction.triggered.connect(
            self._bytetrack_widget.sigReset
        )
        self._video_widget.openAction.triggered.connect(
            self._ocsort_widget.sigReset
        )
        self._video_widget.openAction.triggered.connect(
            self._sort_widget.sigReset
        )
        self._video_widget.openAction.triggered.connect(
            self._csrt_widget.sigReset
        )
        self._video_widget.sigReset.connect(self._bytetrack_widget.sigReset)
        self._video_widget.sigReset.connect(self._ocsort_widget.sigReset)
        self._video_widget.sigReset.connect(self._sort_widget.sigReset)
        self._video_widget.sigReset.connect(self._csrt_widget.sigReset)
        self._video_widget.sigStatusMsg.connect(self.statusMsgSlot)
        self._video_widget.sigOutFilename.connect(self.outFilenameSlot)
        self._seg_grp.triggered.connect(self.switchSegmentation)
        self._track_grp.triggered.connect(self.switchTracking)
        self._quit_action.triggered.connect(self.close)
        self._quit_action.triggered.connect(qw.QApplication.instance().quit)
        self.sigQuit.connect(self._video_widget.sigQuit)
        self.sigQuit.connect(self._yolov11_widget.sigQuit)
        self.sigQuit.connect(self._yolact_widget.sigQuit)
        self.sigQuit.connect(self._seg_widget.sigQuit)
        self.sigQuit.connect(self._lim_widget.sigQuit)
        self.sigQuit.connect(self._bytetrack_widget.sigQuit)
        self.sigQuit.connect(self._ocsort_widget.sigQuit)
        self.sigQuit.connect(self._sort_widget.sigQuit)
        self.sigQuit.connect(self._csrt_widget.sigQuit)

    def makeShortcuts(self):
        self.sc_play = qw.QShortcut(qg.QKeySequence(qc.Qt.Key_Space), self)
        self.sc_play.activated.connect(self._video_widget.playAction.trigger)
        self.sc_refresh = qw.QShortcut(qg.QKeySequence('R'), self)
        self.sc_refresh.activated.connect(
            self._video_widget.refreshAction.trigger
        )
        self.sc_open = qw.QShortcut(qg.QKeySequence('Ctrl+O'), self)
        self.sc_open.activated.connect(self._video_widget.openVideo)
        self.sc_quit = qw.QShortcut(qg.QKeySequence('Ctrl+Q'), self)
        self.sc_quit.activated.connect(self.close)

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
            'Are you sure to clear all saved settings?',
        )
        if button == qw.QMessageBox.NoButton:
            return
        settings.clear()
        settings.sync()

    @qc.pyqtSlot(qw.QAction)
    def switchSegmentation(self, action):
        """Switch segmentation widget between YOLOv11, Yolact, and classical."""
        self._video_widget.pauseVideo()
        # Disconnect all three handlers first, then reconnect only the active one
        for handler in (
            self._yolov11_widget.process,
            self._yolact_widget.process,
            self._seg_widget.sigProcess,
        ):
            try:
                self._video_widget.sigSetFrame.disconnect(handler)
            except TypeError:
                pass
        if action == self._yolov11_action:
            self._video_widget.sigSetFrame.connect(self._yolov11_widget.process)
            self._yolov11_dock.setVisible(True)
            self._yolact_dock.setVisible(False)
            self._seg_dock.setVisible(False)
        elif action == self._yolact_action:
            self._video_widget.sigSetFrame.connect(self._yolact_widget.process)
            self._yolov11_dock.setVisible(False)
            self._yolact_dock.setVisible(True)
            self._seg_dock.setVisible(False)
        else:  # classical segmentation
            self._video_widget.sigSetFrame.connect(self._seg_widget.sigProcess)
            self._yolov11_dock.setVisible(False)
            self._yolact_dock.setVisible(False)
            self._seg_dock.setVisible(True)

    @qc.pyqtSlot(qw.QAction)
    def switchTracking(self, action):
        """Switch tracking between ByteTrack, OC-SORT, SORT, and CSRT."""
        self._video_widget.pauseVideo()
        # CSRT needs the video frame; always disconnect it first
        try:
            self._video_widget.sigSetFrame.disconnect(self._csrt_widget.setFrame)
        except TypeError:
            pass

        self._bytetrack_dock.setVisible(action == self._bytetrack_action)
        self._ocsort_dock.setVisible(action == self._ocsort_action)
        self._sort_dock.setVisible(action == self._sort_action)
        self._csrt_dock.setVisible(action == self._csrt_action)

        if action == self._csrt_action:
            self._video_widget.sigSetFrame.connect(self._csrt_widget.setFrame)

        if action == self._bytetrack_action:
            newhandler = self._bytetrack_widget.sigTrack
        elif action == self._ocsort_action:
            newhandler = self._ocsort_widget.sigTrack
        elif action == self._sort_action:
            newhandler = self._sort_widget.sigTrack
        else:
            newhandler = self._csrt_widget.setBboxes

        if self._yolov11_action.isChecked() or self._yolact_action.isChecked():
            sig = self._lim_widget.sigProcessed
        else:
            sig = self._seg_widget.sigProcessed

        # Disconnect all tracker inputs from sig, then reconnect chosen one
        for old in (self._bytetrack_widget.sigTrack,
                    self._ocsort_widget.sigTrack,
                    self._sort_widget.sigTrack,
                    self._csrt_widget.setBboxes):
            try:
                sig.disconnect(old)
            except TypeError:
                pass
        sig.connect(newhandler)

    @qc.pyqtSlot(str)
    def updateTitle(self, filename: str) -> None:
        self.setWindowTitle(f'Argos:track {filename}')

    @qc.pyqtSlot()
    def saveConfig(self):
        if self._bytetrack_action.isChecked():
            track_method = 'bytetrack'
            min_hits = settings.value('bytetracker/min_hits', 3, type=int)
            max_age = settings.value('bytetracker/max_age', 30, type=int)
            min_dist = settings.value(
                'bytetracker/iou_threshold', 0.3, type=float
            )
        else:
            track_method = 'sort'
            min_hits = settings.value('sortracker/min_hits', 3, type=int)
            max_age = settings.value('sortracker/max_age', 10, type=int)
            min_dist = settings.value('sortracker/min_dist', 0.3, type=float)
        config = {
            'wmin': settings.value('segment/min_width', 10, type=int),
            'wmax': settings.value('segment/max_width', 50, type=int),
            'hmin': settings.value('segment/min_height', 10, type=int),
            'hmax': settings.value('segment/max_height', 100, type=int),
            'track_method': track_method,
            'sort_metric': settings.value(
                'sortracker/metric', 'iou', type=str
            ),
            'min_dist': min_dist,
            'min_hits': min_hits,
            'max_age': max_age,
            'pmin': settings.value('segment/min_pixels', 10, type=int),
            'pmax': settings.value('segment/max_pixels', 1000, type=int),
        }
        if self._seg_dock.isVisible():
            config['blur_width'] = settings.value(
                'segment/blur_width', 21, type=int
            )
            config['blur_sd'] = settings.value(
                'segment/blur_sd', 1.0, type=float
            )
            config['thresh_method'] = settings.value(
                'segment/thresh_method', 'gaussian', type=str
            )
            config['thresh_max'] = settings.value(
                'segment/thresh_max_intensity', 255, type=int
            )
            config['thresh_baseline'] = settings.value(
                'segment/thresh_baseline', type=int
            )
            config['thresh_blocksize'] = settings.value(
                'segment/thresh_blocksize', type=int
            )
            config['thresh_invert'] = settings.value(
                'segment/thresh_invert', True, type=bool
            )
            seg_method = settings.value(
                'segment/method', 'threshold', type=str
            ).lower()
            config['seg_method'] = seg_method
            if seg_method == 'watershed':
                config['dist_thresh'] = int(
                    settings.value(
                        'segment/watershed_distthresh', 3, type=float
                    )
                )
            elif seg_method == 'dbscan':
                config['eps'] = settings.value(
                    'segment/dbscan_eps', 5.0, type=float
                )
                config['min_samples'] = settings.value(
                    'segment/dbscan_minsamples', 10, type=int
                )
        elif self._yolov11_dock.isVisible():
            config['method'] = 'yolov11'
            config['yolov11_weights'] = settings.value(
                'yolov11/weightsfile', '', type=str
            )
            config['score_thresh'] = settings.value(
                'yolov11/score_thresh', 0.25, type=float
            )
            config['top_k'] = settings.value('yolov11/top_k', 10, type=int)
            config['overlap_thresh'] = settings.value(
                'yolov11/overlap_thresh', 0.45, type=float
            )
            config['cuda'] = settings.value('yolov11/cuda', True, type=bool)
        else:  # yolact (legacy)
            config['method'] = 'yolact'
            config['weight'] = settings.value('yolact/weightsfile', type=str)
            config['yconfig'] = settings.value('yolact/configfile', type=str)
            config['overlap_thresh'] = settings.value(
                'yolact/overlap_thresh', 1.0, type=float
            )
            config['score_thresh'] = settings.value(
                'yolact/score_thresh', 0.1, type=float
            )
            config['top_k'] = settings.value('yolact/top_k', 30, type=int)
            config['cuda'] = settings.value('yolact/cuda', True, type=bool)

        directory = settings.value('video/directory', '.')
        fname, _ = qw.QFileDialog.getSaveFileName(
            self,
            'Save configuration as',
            directory,
            filter='yaml (*.yml *.yaml)',
        )
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

    @qc.pyqtSlot(str)
    def outFilenameSlot(self, name):
        self._outfile_label.setText(f'Output: {name}')


if __name__ == '__main__':
    app = qw.QApplication(sys.argv)
    win = ArgosTracker()
    win.setMinimumSize(800, 600)
    win.setWindowTitle('Argos - track animals in video')
    win.showMaximized()
    app.aboutToQuit.connect(win.cleanup)
    win.show()
    sys.exit(app.exec_())
