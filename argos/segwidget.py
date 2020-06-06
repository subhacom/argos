# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-05 11:10 PM
"""Classical image-processing-based segmentation"""
import logging
import numpy as np
import cv2
from sklearn import cluster
from PyQt5 import (
    QtWidgets as qw,
    QtCore as qc
)

from argos import utility as ut


def segment_by_dbscan(binary_img, eps=5, min_samples=10):
    """Use DBSCAN clustering to segment binary image.

    binary_img: binary image, a 2D array containing 0s and 1s
    (obtaind by thresholding an original image).

    eps: the epsilon parameter of DBSCAN.

    min_samples: minimum number of pixels each cluster (object) must
    contain in order to be considered a valid object.

    :return a list of coordinate arrays [arr_0, arr_1, arr_2, ..., arr_n]
            where `n` is the number of clusters (segmented objects),
            `arr_i` is a k x 2 array where each rows are the coordinates
            of the pixels classified as part of object no. `i`.
    """
    indices = np.nonzero(binary_img)
    xy = np.vstack((indices[1], indices[0])).T
    core, labels = cluster.dbscan(xy, eps=eps, min_samples=min_samples,
                                  metric='euclidean', algorithm='auto')
    unique_labels = set(labels)
    unique_labels.discard(-1)  # -1 is the noise label
    return [xy[labels == label] for label in sorted(unique_labels)]


def segment_by_contours(binary_img):
    """Segment binary image by finding contours of contiguous
    nonzero pixels and then filling those contours with an integer
    color value.

    Although, this is also part of the watershed algorithm, for small objects
    that are clearly separable from the background and each other, this
    works equally well.

    binary_img: binary input image (obtained by thresholding grayscale image).

    :return a list of coordinate arrays [arr_0, arr_1, arr_2, ..., arr_n]
            where `n` is the number of clusters (segmented objects),
            `arr_i` is a k x 2 array where each rows are the coordinates
            of the pixels classified as part of object no. `i`.
    """
    contours, hierarchy = cv2.findContours(binary_img,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    logging.debug(f'Segmented {len(contours)} objects.')
    segmented = np.zeros(binary_img.shape, dtype=np.int32)
    for ii, contour in enumerate(contours):
        cv2.drawContours(segmented, [contour], -1, thickness=cv2.FILLED,
                         color=ii + 1)
    unique_labels = set(segmented.flat)
    unique_labels.discard(0)
    ret = [np.argwhere(segmented == label) for label in sorted(unique_labels)]
    # Fast swapping of y and x - see answer by blax here:
    # https://stackoverflow.com/questions/4857927/swapping-columns-in-a-numpy-array
    for points in ret:
        logging.debug('***** %r', points.shape)
        points[:, 0], points[:, 1] = points[:, 1], points[:, 0].copy()
    logging.debug(f'Returning {len(ret)} point-arrays.')
    if len(ret) > 0:
        logging.debug(f'First points-array of shape {ret[0].shape}')
    return ret



def extract_valid(points_list, pmin, pmax, wmin, wmax, hmin, hmax):
    """From a list of coordinate arrays for object pixels find the ones that
    is between `pmin` and `pmax` pixels, `wmin` and `wmax` width, and
    `hmin` and `hmax` height. I arbitrarily take the length of the smaller
    side of the minimum bounding rotated-rectangle as width and the larger
    as height.
    """
    logging.debug(f'Parameters: pmin {pmin}, pmax {pmax}, wmin {wmin}, '
                  f'wmax {wmax}, hmin {hmin}, hmax {hmax}')
    mr_size = np.array([cv2.minAreaRect(points)[1] for points in points_list])
    mr_size.sort(axis=1)
    p_size = np.array([len(points) for points in points_list])
    good = (p_size >= pmin) & (p_size < pmax) \
           & (mr_size[:, 0] >= wmin) & (mr_size[:, 0] < wmax) \
           & (mr_size[:, 1] >= hmin) & (mr_size[:, 1] < hmax)
    good = np.flatnonzero(good)
    logging.debug(f'From {len(points_list)} indices fitting size conds: {good}')
    return [points_list[ii] for ii in good]


class SegWorker(qc.QObject):
    sigProcessed = qc.pyqtSignal(np.ndarray, int)

    def __init__(self):
        super(SegWorker, self).__init__()
        # blurring parameters
        self.kernel_width = 7
        self.kernel_sd = 1.0
        # threshold parameters
        self.thresh_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        self.invert = True
        self.block_size = 41
        self.max_intensity = 255
        # this baseline value is subtracted from the neighborhood weighted mean
        self.baseline =10
        # segmentation method
        self.seg_method = ut.SegmentationMethod.threshold
        self.dbscan_eps = 5
        self.dbscan_min_samples =10
        # cleanup params
        self.pmin = 10
        self.pmax = 500
        self.wmin = 20
        self.wmax = 50
        self.hmin = 50
        self.hmax = 200

    @qc.pyqtSlot(int)
    def setBlurWidth(self, width: int) -> None:
        self.kernel_width = width

    @qc.pyqtSlot(float)
    def setBlurSigma(self, sigma: float) -> None:
        self.kernel_sd = sigma

    @qc.pyqtSlot(int)
    def setInvertThreshold(self, invert: int) -> None:
        self.invert = invert

    @qc.pyqtSlot(int)
    def setThresholdMethod(self, method: int) -> None:
        self.thresh_method = method

    @qc.pyqtSlot(int)
    def setMaxIntensity(self, value: int) -> None:
        self.max_intensity = value

    @qc.pyqtSlot(int)
    def setBaseline(self, value: int) -> None:
        self.baseline = value

    @qc.pyqtSlot(ut.SegmentationMethod)
    def setSegmentationMethod(self, method: ut.SegmentationMethod) -> None:
        self.seg_method = method

    @qc.pyqtSlot(float)
    def setEpsDBSCAN(self, value: int) -> None:
        self.dbscan_eps = value

    @qc.pyqtSlot(int)
    def setMinSamplesDBSCAN(self, value: int) -> None:
        self.dbscan_min_samples = value

    @qc.pyqtSlot(int)
    def setMinPixels(self, value: int) -> None:
        self.pmin = value

    @qc.pyqtSlot(int)
    def setMaxPixels(self, value: int) -> None:
        self.pmax = value

    @qc.pyqtSlot(int)
    def setMinWidth(self, value: int) -> None:
        self.wmin = value

    @qc.pyqtSlot(int)
    def setMaxWidth(self, value: int) -> None:
        self.wmax = value

    @qc.pyqtSlot(int)
    def setMinHeight(self, value: int) -> None:
        self.hmin = value

    @qc.pyqtSlot(int)
    def setMaxHeight(self, value: int) -> None:
        self.hmax = value

    @qc.pyqtSlot(np.ndarray, int)
    def process(self, image: np.ndarray, pos: int) -> None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(
            image,
            ksize=(self.kernel_width, self.kernel_width),
            sigmaX=self.kernel_sd)
        if self.invert:
            thresh_type = cv2.THRESH_BINARY_INV
        else:
            thresh_type = cv2.THRESH_BINARY
        image = cv2.adaptiveThreshold(image, maxValue=self.max_intensity,
                                      adaptiveMethod=self.thresh_method,
                                      thresholdType=thresh_type,
                                      blockSize=self.block_size,
                                      C=self.baseline)
        if self.seg_method == ut.SegmentationMethod.threshold:
            seg = segment_by_contours(image)
        elif self.seg_method == ut.SegmentationMethod.dbscan:
            seg = segment_by_dbscan(image, self.dbscan_eps,
                                    self.dbscan_min_samples)
        seg = extract_valid(seg, self.pmin, self.pmax, self.wmin, self.wmax,
                            self.hmin, self.hmax)
        bboxes = [cv2.boundingRect(points) for points in seg]
        self.sigProcessed.emit(np.array(bboxes), pos)


class SegWidget(qw.QWidget):
    # pass on the signal from worker
    sigProcessed = qc.pyqtSignal(np.ndarray, int)
    # pass on the image to worker
    sigProcess = qc.pyqtSignal(np.ndarray, int)

    sigThreshMethod = qc.pyqtSignal(int)
    sigSegMethod = qc.pyqtSignal(ut.SegmentationMethod)

    sigQuit = qc.pyqtSignal()


    def __init__(self, *args, **kwargs):
        super(SegWidget, self).__init__(*args, **kwargs)
        self.worker = SegWorker()
        layout = qw.QFormLayout()
        self._blur_width_label = qw.QLabel('Blur width')
        self._blur_width_edit = qw.QSpinBox()
        self._blur_width_edit.setRange(1, 100)
        self._blur_width_edit.setValue(self.worker.kernel_width)
        layout.addRow(self._blur_width_label, self._blur_width_edit)
        self._blur_sd_label = qw.QLabel('Blur sd')
        self._blur_sd_edit = qw.QDoubleSpinBox()
        self._blur_sd_edit.setRange(1, 100)
        self._blur_sd_edit.setValue(self.worker.kernel_sd)
        layout.addRow(self._blur_sd_label, self._blur_sd_edit)
        self._invert_label = qw.QLabel('Invert thresholding')
        self._invert_check = qw.QCheckBox()
        self._invert_check.setChecked(self.worker.invert)
        layout.addRow(self._invert_label, self._invert_check)
        self._thresh_label = qw.QLabel('Thresholding method')
        self._thresh_method = qw.QComboBox()
        self._thresh_method.addItems(['Adaptive Gaussian', 'Adaptive Mean'])
        layout.addRow(self._thresh_label, self._thresh_method)
        self._maxint_label = qw.QLabel('Threshold maximum intensity')
        self._maxint_edit = qw.QSpinBox()
        self._maxint_edit.setRange(0, 255)
        self._maxint_edit.setValue(self.worker.max_intensity)
        layout.addRow(self._maxint_label, self._maxint_edit)
        self._baseline_label = qw.QLabel('Threshold baseline')
        self._baseline_edit = qw.QSpinBox()
        self._baseline_edit.setRange(0, 255)
        self._baseline_edit.setValue(self.worker.baseline)
        layout.addRow(self._baseline_label, self._baseline_edit)
        self._seg_label = qw.QLabel('Segmentation method')
        self._seg_method = qw.QComboBox()
        self._seg_method.addItems(['Threshold', 'DBSCAN'])

        layout.addRow(self._seg_label, self._seg_method)
        self._dbscan_minsamples_label = qw.QLabel('DBSCAN minimum samples')
        self._dbscan_minsamples = qw.QSpinBox()
        self._dbscan_minsamples.setRange(1, 1000)
        self._dbscan_minsamples.setValue(self.worker.dbscan_min_samples)
        layout.addRow(self._dbscan_minsamples_label, self._dbscan_minsamples)
        self._dbscan_eps_label = qw.QLabel('DBSCAN epsilon')
        self._dbscan_eps = qw.QDoubleSpinBox()
        self._dbscan_eps.setRange(1, 100)
        self._dbscan_eps.setValue(self.worker.dbscan_eps)
        layout.addRow(self._dbscan_eps_label, self._dbscan_eps)
        self._pmin_label = qw.QLabel('Minimum pixels')
        self._pmin_edit = qw.QSpinBox()
        self._pmin_edit.setRange(1, 1000)
        self._pmin_edit.setValue(self.worker.pmin)
        layout.addRow(self._pmin_label, self._pmin_edit)
        self._pmax_label = qw.QLabel('Maximum pixels')
        self._pmax_edit = qw.QSpinBox()
        self._pmax_edit.setRange(1, 1000)
        self._pmax_edit.setValue(self.worker.pmax)
        layout.addRow(self._pmax_label, self._pmax_edit)
        self._wmin_label = qw.QLabel('Minimum width')
        self._wmin_edit = qw.QSpinBox()
        self._wmin_edit.setRange(1, 1000)
        self._wmin_edit.setValue(self.worker.wmin)
        layout.addRow(self._wmin_label, self._wmin_edit)
        self._wmax_label = qw.QLabel('Maximum width')
        self._wmax_edit = qw.QSpinBox()
        self._wmax_edit.setRange(1, 1000)
        self._wmax_edit.setValue(self.worker.wmax)
        layout.addRow(self._wmax_label, self._wmax_edit)
        self._hmin_label = qw.QLabel('Minimum length')
        self._hmin_edit = qw.QSpinBox()
        self._hmin_edit.setRange(1, 1000)
        self._hmin_edit.setValue(self.worker.hmin)
        layout.addRow(self._hmin_label, self._hmin_edit)
        self._hmax_label = qw.QLabel('Maximum length')
        self._hmax_edit = qw.QSpinBox()
        self._hmax_edit.setRange(1, 1000)
        self._hmax_edit.setValue(self.worker.hmax)
        layout.addRow(self._hmax_label, self._hmax_edit)
        self.setLayout(layout)
        # Housekeeping for convenience
        self._seg_param_widgets = {
            'Threshold': [],
            'DBSCAN': [self._dbscan_eps, self._dbscan_minsamples,
                       self._dbscan_minsamples_label, self._dbscan_eps_label]

        }
        if self.worker.seg_method == ut.SegmentationMethod.threshold:
            self._seg_method.setCurrentText('Threshold')
        elif self.worker.seg_method == ut.SegmentationMethod.dbscan:
            self._seg_method.setCurrentText('DBSCAN')
        self.setSegmentationMethod(self._seg_method.currentText())
        ###################################
        # Threading
        self.thread = qc.QThread()
        self.worker.moveToThread(self.thread)
        ####################################
        # Connections
        self._blur_width_edit.valueChanged.connect(self.worker.setBlurWidth)
        self._blur_sd_edit.valueChanged.connect(self.worker.setBlurSigma)
        self._invert_check.stateChanged.connect(self.worker.setInvertThreshold)
        self._thresh_method.currentTextChanged.connect(self.setThresholdMethod)
        self._maxint_edit.valueChanged.connect(self.worker.setMaxIntensity)
        self._baseline_edit.valueChanged.connect(self.worker.setBaseline)
        self._seg_method.currentTextChanged.connect(self.setSegmentationMethod)
        self.sigThreshMethod.connect(self.worker.setThresholdMethod)
        self.sigSegMethod.connect(self.worker.setSegmentationMethod)
        self._dbscan_minsamples.valueChanged.connect(
            self.worker.setMinSamplesDBSCAN)
        self._dbscan_eps.valueChanged.connect(
            self.worker.setEpsDBSCAN
        )
        self._pmin_edit.valueChanged.connect(self.worker.setMinPixels)
        self._pmax_edit.valueChanged.connect(self.worker.setMaxPixels)
        self._wmin_edit.valueChanged.connect(self.worker.setMinWidth)
        self._wmax_edit.valueChanged.connect(self.worker.setMaxWidth)
        self._hmin_edit.valueChanged.connect(self.worker.setMinHeight)
        self._hmax_edit.valueChanged.connect(self.worker.setMaxHeight)
        self.sigProcess.connect(self.worker.process)
        self.worker.sigProcessed.connect(self.sigProcessed)
        ###################
        # Thread setup
        self.sigQuit.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @qc.pyqtSlot(str)
    def setThresholdMethod(self, text) -> None:
        if text == 'Adaptive Gaussian':
            self.sigThreshMethod.emit(cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        elif text == 'Adaptive Mean':
            self.sigThreshMethod.emit(cv2.ADAPTIVE_THRESH_MEAN_C)

    @qc.pyqtSlot(str)
    def setSegmentationMethod(self, text: str) -> None:
        for key, widgets in self._seg_param_widgets.items():
            if text == key:
                [widget.setVisible(True) for widget in widgets]
            else:
                [widget.setVisible(False) for widget in widgets]
        if text == 'Threshold':
            self.sigSegMethod.emit(ut.SegmentationMethod.threshold)
        elif text == 'DBSCAN':
            self.sigSegMethod.emit(ut.SegmentationMethod.dbscan)
        else:
            raise NotImplementedError(f'{text} method not available')



