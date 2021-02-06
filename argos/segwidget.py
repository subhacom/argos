# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-05 11:10 PM

"""Classical image-processing-based segmentation"""

import logging
from collections import OrderedDict
import time
import numpy as np
import cv2
from PyQt5 import (
    QtWidgets as qw,
    QtCore as qc,
    QtGui as qg
)

import argos.constants as consts
from argos import utility as ut
from argos.frameview import FrameView
from argos.segment import (
    segment_by_dbscan,
    segment_by_contours,
    segment_by_contour_bbox,
    segment_by_watershed,
    extract_valid,
    get_bounding_poly
)

settings = ut.init()

segstep_dict = OrderedDict([
    ('Final', consts.SegStep.final),
    ('Blurred', consts.SegStep.blur),
    ('Thresholded', consts.SegStep.threshold),
    ('Segmented', consts.SegStep.segmented),
    ('Filtered', consts.SegStep.filtered),
])

segmethod_dict = OrderedDict([
    ('Threshold', consts.SegmentationMethod.threshold),
    ('Contour', consts.SegmentationMethod.contour),
    ('Watershed', consts.SegmentationMethod.watershed),
    ('DBSCAN', consts.SegmentationMethod.dbscan)
])

outline_dict = OrderedDict([
    ('bbox', ut.OutlineStyle.bbox),
    ('minrect', ut.OutlineStyle.minrect),
    ('contour', ut.OutlineStyle.contour),
    ('fill', ut.OutlineStyle.fill)
])


class SegWorker(qc.QObject):
    """Worker class for carrying out segmentation.

    This class provides access to three different segmentation methods:

    thresholding: where the image is converted into grayscale and then an
    adaptive thresholding algorithm is is applied to obtain blobs.

    DBSCAN: where the binary pixels are spatially clustered using the DBSCAN
    algorithm and the valid clusters constitute segmented objects.

    Watershed: the classic Watershed algorithm.

    Attributes
    ----------
    mode: argos.utility.OutlineStyle
        If mode is bbox, the array of bounding boxes of the segmented objects is
        sent out via ``sigProcessed``. If contour, an array containing the
        points forming the contour of each object is sent out, each row of the
        form (x, y, index) where x, y is a point on the contour, and index is
        the index of the segmented object.
    kernel_width: int
        Width of Gaussian kernel used in blurring the image.
    kernel_sd: float
        Standard deviation of the Gaussian used in blurring the image.
    thresh_method: {cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.ADAPTIVE_THRESH_MEAN_C}
        Method to use for adaptive thresholding the grayscaled, blurred image.
    invert: bool
        Inverse thresholding (threshold below, instead of above).
    block_size:
        Block size for adaptive thresholding.
    max_intensity: int
        Maximum intensity values to set for pixels crossing threshold value.
    baseline: int
        baseline value for adaptive thresholding.
    seg_method: {argos.utility.SegmentationMethod.threshold,
                argos.utility.SegmentationMethod.dbscan,
                argos.utility.SegmentationMethod.watershed}
        Segmentation method, choice between adaptive thresholding, DBSCAN and
        Watershed algorithm.
    dbscan_eps: float
        epsilon parameter in DBSCAN algorithm.
    dbscan_min_samples: int
        Minimum number of samples in each cluster in DBSCAN algorithm.
    wdist_thresh: float
        Distance threshold for finding core pixels before applying watershed
        algorithm.
    pmin: int
    pmax: int
    wmin: int
    wmax: int
    hmin: int
    hmax: int
        See extract_valid function above.
    intermediate: argos.utility.SegStep
        Intermediate step whose output should be emitted via
        ``sigIntermediate``. If ``argos.utility.SegStep.final``, then
        intermediate result is not emitted. Otherwise this can be used for
        deciding best parameters for various steps in the segmentation process.

    cmap: int, cv2 colormap
        Colormap for converting label image from grayscale to RGB for better
        visualization of intermediate segmentation result.

    """
    # bboxes of segmented objects and frame no.
    sigProcessed = qc.pyqtSignal(np.ndarray, int)
    # outlines of segmented objects and frame no.
    sigSegPolygons = qc.pyqtSignal(dict, int)
    # intermediate processed image and frame no.
    sigIntermediate = qc.pyqtSignal(np.ndarray, int)

    def __init__(self, mode=ut.OutlineStyle.bbox):
        super(SegWorker, self).__init__()
        self.outline_style = mode
        # blurring parameters
        self.kernel_width = 7
        self.kernel_sd = 1.0
        # threshold parameters
        self.thresh_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        self.invert = True
        self.block_size = 41
        self.max_intensity = 255
        # this baseline value is subtracted from the neighborhood weighted mean
        self.baseline = 10
        # segmentation method
        self.seg_method = consts.SegmentationMethod.threshold
        #  DBSCAN parameters
        self.dbscan_eps = 5
        self.dbscan_min_samples = 10
        # Watershed algorithm - distance threshold
        self.wdist_thresh = 3.0
        # cleanup params
        self.pmin = 10
        self.pmax = 500
        self.wmin = 20
        self.wmax = 50
        self.hmin = 50
        self.hmax = 200
        self.roi = None
        self.intermediate = consts.SegStep.final
        self.cmap = cv2.COLORMAP_JET

    @qc.pyqtSlot(ut.OutlineStyle)
    def setOutlineStyle(self, mode: ut.OutlineStyle) -> None:
        self.outline_style = mode

    @qc.pyqtSlot(consts.SegStep)
    def setIntermediateOutput(self, step: consts.SegStep) -> None:
        self.intermediate = step

    @qc.pyqtSlot(int)
    def setBlurWidth(self, width: int) -> None:
        if width % 2 == 0:
            width += 1
        settings.setValue('segment/blur_width', width)
        self.kernel_width = width

    @qc.pyqtSlot(float)
    def setBlurSigma(self, sigma: float) -> None:
        settings.setValue('segment/blur_sd', sigma)
        self.kernel_sd = sigma

    @qc.pyqtSlot(int)
    def setInvertThreshold(self, invert: int) -> None:
        settings.setValue('segment/thresh_invert', invert)
        self.invert = invert

    @qc.pyqtSlot(int)
    def setThresholdMethod(self, method: int) -> None:
        if method == cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
            settings.setValue('segment/thresh_method', 'gaussian')
        else:
            settings.setValue('segment/thresh_method', 'mean')
        self.thresh_method = method

    @qc.pyqtSlot(int)
    def setMaxIntensity(self, value: int) -> None:
        settings.setValue('segment/thresh_max_intensity',
                          value)
        self.max_intensity = value

    @qc.pyqtSlot(int)
    def setBaseline(self, value: int) -> None:
        settings.setValue('segment/thresh_baseline', value)
        self.baseline = value

    @qc.pyqtSlot(int)
    def setBlockSize(self, value: int) -> None:
        if value % 2 == 0:
            self.block_size = value + 1
        else:
            self.block_size = value
        settings.setValue('segment/thresh_blocksize', self.block_size)

    @qc.pyqtSlot(consts.SegmentationMethod)
    def setSegmentationMethod(self,
                              method: consts.SegmentationMethod) -> None:
        if method == consts.SegmentationMethod.dbscan:
            settings.setValue('segment/method', 'DBSCAN')
        elif method == consts.SegmentationMethod.watershed:
            settings.setValue('segment/method', 'Watershed')
        else:
            settings.setValue('segment/method', 'Threshold')

        self.seg_method = method

    @qc.pyqtSlot(float)
    def setEpsDBSCAN(self, value: int) -> None:
        settings.setValue('segment/dbscan_eps', value)
        self.dbscan_eps = value

    @qc.pyqtSlot(int)
    def setMinSamplesDBSCAN(self, value: int) -> None:
        settings.setValue('segment/dbscan_minsamples', value)
        self.dbscan_min_samples = value

    @qc.pyqtSlot(float)
    def setDistThreshWatershed(self, value: float) -> None:
        settings.setValue('segment/watershed_distthresh', value)
        self.wdist_thresh = value

    @qc.pyqtSlot(int)
    def setMinPixels(self, value: int) -> None:
        settings.setValue('segment/min_pixels', value)
        self.pmin = value

    @qc.pyqtSlot(int)
    def setMaxPixels(self, value: int) -> None:
        settings.setValue('segment/max_pixels', value)
        self.pmax = value

    @qc.pyqtSlot(int)
    def setMinWidth(self, value: int) -> None:
        settings.setValue('segment/min_width', value)
        self.wmin = value

    @qc.pyqtSlot(int)
    def setMaxWidth(self, value: int) -> None:
        settings.setValue('segment/max_width', value)
        self.wmax = value

    @qc.pyqtSlot(int)
    def setMinHeight(self, value: int) -> None:
        settings.setValue('segment/min_height', value)
        self.hmin = value

    @qc.pyqtSlot(int)
    def setMaxHeight(self, value: int) -> None:
        settings.setValue('segment/max_height', value)
        self.hmax = value

    @qc.pyqtSlot(qg.QPolygonF)
    def setRoi(self, roi: qg.QPolygonF):
        self.roi = roi

    @qc.pyqtSlot()
    def resetRoi(self):
        self.roi = None

    @qc.pyqtSlot(np.ndarray, int)
    def process(self, image: np.ndarray, pos: int) -> None:
        """Segment image

        Parameters
        ----------
        image: np.ndarray
            Image to be segmented (an array in OpenCV's BGR format)
        pos: int
            Frame no. for this image. Not used in processing, but emitted along
            with processed data so downstream components can associate the
            result with frame position.

        Notes
        -----
        Emits array of the bounding boxes of segmented objects via
        ``sigProcessed`` signal.

        Unless ``intermediate`` is not ``argos.utility.SegStep.final``, emits
        intermediate result via ``sigIntermediate``. The intermediate result
        is a grayscale or binary image after the blur and threshold steps
        respectively, and for ``argos.utility.Segmented``, a pseudo-color image
        with different color values for each segmented object.
        """
        _ts = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(
            gray,
            ksize=(self.kernel_width, self.kernel_width),
            sigmaX=self.kernel_sd)
        if self.intermediate == consts.SegStep.blur:
            self.sigIntermediate.emit(gray, pos)
        if self.invert:
            thresh_type = cv2.THRESH_BINARY_INV
        else:
            thresh_type = cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(gray,
                                       maxValue=self.max_intensity,
                                       adaptiveMethod=self.thresh_method,
                                       thresholdType=thresh_type,
                                       blockSize=self.block_size,
                                       C=self.baseline)
        if self.intermediate == consts.SegStep.threshold:
            self.sigIntermediate.emit(binary, pos)
        if self.seg_method == consts.SegmentationMethod.threshold:
            seg = segment_by_contour_bbox(binary)
        elif self.seg_method == consts.SegmentationMethod.contour:
            seg = segment_by_contours(binary)
        elif self.seg_method == consts.SegmentationMethod.dbscan:
            seg = segment_by_dbscan(binary, self.dbscan_eps,
                                    self.dbscan_min_samples)
        elif self.seg_method == consts.SegmentationMethod.watershed:
            seg = segment_by_watershed(binary, image,
                                                  self.wdist_thresh)
        if self.intermediate == consts.SegStep.segmented:
            for ii, points in enumerate(seg):
                binary[points[:, 1], points[:, 0]] = ii + 1
            self.sigIntermediate.emit(
                cv2.applyColorMap(binary, self.cmap),
                pos)
        seg = extract_valid(seg, self.pmin, self.pmax, self.wmin, self.wmax,
                            self.hmin, self.hmax, roi=self.roi)
        if self.intermediate == consts.SegStep.filtered:
            for ii, points in enumerate(seg):
                binary[points[:, 1], points[:, 0]] = ii + 1
            self.sigIntermediate.emit(
                cv2.applyColorMap(binary, self.cmap),
                pos)
        bboxes = [cv2.boundingRect(points) for points in seg]
        if self.outline_style == ut.OutlineStyle.bbox:
            self.sigProcessed.emit(np.array(bboxes), pos)
            # logging.debug(f'Emitted bboxes for frame {pos}: {bboxes}')
        elif self.outline_style == ut.OutlineStyle.contour:
            if self.seg_method == consts.SegmentationMethod.threshold:
                contours = {ii: contour for ii, contour in enumerate(seg)}
            else:
                contours = get_bounding_poly(seg, ut.OutlineStyle.contour)
                contours = {ii: contour for ii, contour in enumerate(contours)}
            self.sigSegPolygons.emit(contours, pos)

        elif self.outline_style == ut.OutlineStyle.minrect:
            minrects = [cv2.boxPoints(cv2.minAreaRect(points)) for points in
                        seg]
            minrects = {ii: box for ii, box in enumerate(minrects)}
            self.sigSegPolygons.emit(minrects, pos)
        elif self.outline_style == ut.OutlineStyle.fill:
            filled = {ii: points for ii, points in enumerate(seg)}
            self.sigSegPolygons.emit(filled, pos)
        _dt = time.perf_counter() - _ts
        logging.debug(f'{__name__}.{self.__class__.__name__}.process: Runtime: {_dt}s')

class SegWidget(qw.QWidget):
    """Widget for classical segmentation.

    Provides controls for parameters used in segmentation.
    """
    # pass on the signal from worker
    sigProcessed = qc.pyqtSignal(np.ndarray, int)
    # pass on the signal from worker
    sigSegPolygons = qc.pyqtSignal(dict, int)
    # pass on the image to worker
    sigProcess = qc.pyqtSignal(np.ndarray, int)

    sigSetOutlineStyle = qc.pyqtSignal(ut.OutlineStyle)

    sigThreshMethod = qc.pyqtSignal(int)
    sigSegMethod = qc.pyqtSignal(consts.SegmentationMethod)
    sigIntermediateOutput = qc.pyqtSignal(consts.SegStep)

    sigQuit = qc.pyqtSignal()

    setWmin = qc.pyqtSignal(int)
    setWmax = qc.pyqtSignal(int)
    setHmin = qc.pyqtSignal(int)
    setHmax = qc.pyqtSignal(int)
    setRoi = qc.pyqtSignal(qg.QPolygonF)
    resetRoi = qc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(SegWidget, self).__init__(*args, **kwargs)
        self.worker = SegWorker()
        layout = qw.QFormLayout()
        self._blur_width_label = qw.QLabel('Blur width')
        self._blur_width_edit = qw.QSpinBox()
        self._blur_width_edit.setRange(1, 999)
        self._blur_width_edit.setSingleStep(2)
        value = settings.value('segment/blur_width', self.worker.kernel_width,
                               type=int)
        self._blur_width_edit.setValue(value)
        self.worker.kernel_width = value
        layout.addRow(self._blur_width_label, self._blur_width_edit)
        self._blur_sd_label = qw.QLabel('Blur sd')
        self._blur_sd_edit = qw.QDoubleSpinBox()
        self._blur_sd_edit.setRange(1, 100)
        value = settings.value('segment/blur_sd', self.worker.kernel_sd,
                               type=float)
        self._blur_sd_edit.setValue(value)
        self.worker.kernel_sd = value
        layout.addRow(self._blur_sd_label, self._blur_sd_edit)
        self._invert_label = qw.QLabel('Invert thresholding')
        self._invert_check = qw.QCheckBox()
        self._invert_check.setToolTip('Check this if the objects of interest'
                                      ' are darker than background.')
        value = settings.value('segment/thresh_invert', self.worker.invert,
                               type=bool)
        self._invert_check.setChecked(value)
        self.worker.invert = value
        layout.addRow(self._invert_label, self._invert_check)
        self._thresh_label = qw.QLabel('Thresholding method')
        self._thresh_method = qw.QComboBox()
        self._thresh_method.addItems(['Adaptive Gaussian', 'Adaptive Mean'])
        layout.addRow(self._thresh_label, self._thresh_method)
        self._maxint_label = qw.QLabel('Threshold maximum intensity')
        self._maxint_edit = qw.QSpinBox()
        self._maxint_edit.setRange(0, 255)
        value = settings.value('segment/thresh_max_intensity',
                               self.worker.max_intensity, type=int)
        self.worker.max_intensity = value
        self._maxint_edit.setValue(value)
        layout.addRow(self._maxint_label, self._maxint_edit)
        self._baseline_label = qw.QLabel('Threshold baseline')
        self._baseline_edit = qw.QSpinBox()
        self._baseline_edit.setRange(0, 255)
        self._baseline_edit.setToolTip('This value is subtracted from the'
                                       ' (weighted) mean pixel value in the'
                                       ' neighborhood of a pixel to get the'
                                       ' threshold value at that pixel.')
        value = settings.value('segment/thresh_baseline', self.worker.baseline,
                               type=int)
        self._baseline_edit.setValue(value)
        self.worker.baseline = value
        layout.addRow(self._baseline_label, self._baseline_edit)
        self._blocksize_label = qw.QLabel('Thresholding block size')
        self._blocksize_edit = qw.QSpinBox()
        self._blocksize_edit.setRange(3, 501)
        self._blocksize_edit.setSingleStep(2)
        self._blocksize_edit.setToolTip('Adapaive thresholding block size. Size'
                                        ' of neighborhood for computing local'
                                        ' threshold at each pixel. Should be'
                                        ' odd number >= 3.')
        value = settings.value('segment/thresh_blocksize',
                               self.worker.block_size,
                               type=int)
        if value % 2 == 0:
            value += 1
        self.worker.block_size = value
        self._blocksize_edit.setValue(value)
        layout.addRow(self._blocksize_label, self._blocksize_edit)
        self._seg_label = qw.QLabel('Segmentation method')
        self._seg_method = qw.QComboBox()
        self._seg_method.addItems(segmethod_dict.keys())

        layout.addRow(self._seg_label, self._seg_method)
        self._dbscan_minsamples_label = qw.QLabel('DBSCAN minimum samples')
        self._dbscan_minsamples = qw.QSpinBox()
        self._dbscan_minsamples.setRange(1, 10000)
        value = settings.value('segment/dbscan_minsamples',
                               self.worker.dbscan_min_samples,
                               type=int)
        self._dbscan_minsamples.setValue(value)
        self.worker.dbscan_min_samples = value
        layout.addRow(self._dbscan_minsamples_label, self._dbscan_minsamples)
        self._dbscan_eps_label = qw.QLabel('DBSCAN epsilon')
        self._dbscan_eps = qw.QDoubleSpinBox()
        self._dbscan_eps.setRange(0.1, 100)
        try:
            # setStepType was added in Qt v 5.12 only
            self._dbscan_eps.setStepType(
                qw.QAbstractSpinBox.AdaptiveDecimalStepType)
        except AttributeError:
            pass  # Avoid problem with older Qt versions
        value = settings.value('segment/dbscan_eps',
                               self.worker.dbscan_eps,
                               type=float)
        self._dbscan_eps.setValue(value)
        self.worker.dbscan_eps = value
        layout.addRow(self._dbscan_eps_label, self._dbscan_eps)
        self._pmin_label = qw.QLabel('Minimum pixels')
        self._pmin_edit = qw.QSpinBox()
        self._pmin_edit.setRange(1, 10000)
        value = settings.value('segment/min_pixels',
                               self.worker.pmin,
                               type=int)
        self._pmin_edit.setValue(value)
        self.worker.pmin = value
        layout.addRow(self._pmin_label, self._pmin_edit)
        self._pmax_label = qw.QLabel('Maximum pixels')
        self._pmax_edit = qw.QSpinBox()
        self._pmax_edit.setRange(1, 10000)
        value = settings.value('segment/max_pixels',
                               self.worker.pmax,
                               type=int)
        self._pmax_edit.setValue(value)
        self.worker.pmax = value
        layout.addRow(self._pmax_label, self._pmax_edit)
        # self._wmin_label = qw.QLabel('Minimum width')
        # self._wmin_edit = qw.QSpinBox()
        # self._wmin_edit.setRange(1, 1000)
        value = settings.value('segment/min_width',
                               self.worker.wmin,
                               type=int)
        # self._wmin_edit.setValue(value)
        self.worker.wmin = value
        # layout.addRow(self._wmin_label, self._wmin_edit)
        # self._wmax_label = qw.QLabel('Maximum width')
        # self._wmax_edit = qw.QSpinBox()
        # self._wmax_edit.setRange(1, 1000)
        value = settings.value('segment/max_width',
                               self.worker.wmax,
                               type=int)
        # self._wmax_edit.setValue(value)
        self.worker.wmax = value
        # layout.addRow(self._wmax_label, self._wmax_edit)
        # self._hmin_label = qw.QLabel('Minimum length')
        # self._hmin_edit = qw.QSpinBox()
        # self._hmin_edit.setRange(1, 1000)
        value = settings.value('segment/min_height',
                               self.worker.hmin,
                               type=int)
        # self._hmin_edit.setValue(value)
        self.worker.hmin = value
        # layout.addRow(self._hmin_label, self._hmin_edit)
        # self._hmax_label = qw.QLabel('Maximum length')
        # self._hmax_edit = qw.QSpinBox()
        # self._hmax_edit.setRange(1, 1000)
        value = settings.value('segment/max_height',
                               self.worker.hmax,
                               type=int)
        # self._hmax_edit.setValue(value)
        self.worker.hmax = value
        # layout.addRow(self._hmax_label, self._hmax_edit)
        self._wdist_label = qw.QLabel('Distance threshold')
        self._wdist = qw.QDoubleSpinBox()
        self._wdist.setRange(0, 10)
        try:
            self._wdist.setStepType(qw.QAbstractSpinBox.AdaptiveDecimalStepType)
        except AttributeError:
            pass
        self._wdist.setSingleStep(0.1)
        value = settings.value('segment/watershed_distthresh',
                               self.worker.wdist_thresh,
                               type=float)
        self._wdist.setValue(value)
        self.worker.wdist_thresh = value
        self._wdist.setToolTip('Distance threshold for Watershed segmentation.'
                               ' This is used for finding foreground areas in'
                               ' the thresholded image. The pixels which are at'
                               ' least this much (in pixel units) away from all'
                               ' background pixels, are considered to be part'
                               ' of the foreground.\n'
                               ' If set between 0 and 1, it is assumed to be'
                               ' the fraction of maximum of the distances of'
                               ' each pixel from a zero-valued pixel in the'
                               ' thresholded image.')
        layout.addRow(self._wdist_label, self._wdist)
        self._intermediate_label = qw.QLabel('Show intermediate steps')
        self._intermediate_combo = qw.QComboBox()
        self._intermediate_combo.addItems(segstep_dict.keys())
        layout.addRow(self._intermediate_label, self._intermediate_combo)

        self._outline_label = qw.QLabel('Boundary style')
        self.outlineCombo = qw.QComboBox()
        self.outlineCombo.addItems(list(outline_dict.keys()))
        self.outlineCombo.currentTextChanged.connect(self.setOutlineStyle)
        layout.addRow(self._outline_label, self.outlineCombo)

        self.setLayout(layout)

        self._intermediate_win = FrameView()
        self.resetRoi.connect(self._intermediate_win.resetArenaAction.trigger)
        self._intermediate_win.setWindowFlag(
            qc.Qt.Window + qc.Qt.WindowStaysOnTopHint)
        self._intermediate_win.hide()
        # Housekeeping - widgets to be shown or hidden based on choice of method
        self._seg_param_widgets = {
            'Threshold': [],
            'DBSCAN': [self._dbscan_eps, self._dbscan_minsamples,
                       self._dbscan_minsamples_label, self._dbscan_eps_label],
            'Watershed': [self._wdist_label, self._wdist]
        }

        value = settings.value('segment/method', 'Threshold', type=str)
        self.worker.seg_method = segmethod_dict[value]
        self._seg_method.setCurrentText(value)
        self.setSegmentationMethod(value)
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
        self._blocksize_edit.valueChanged.connect(self.worker.setBlockSize)
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
        # self._wmin_edit.valueChanged.connect(self.worker.setMinWidth)
        # self._wmax_edit.valueChanged.connect(self.worker.setMaxWidth)
        # self._hmin_edit.valueChanged.connect(self.worker.setMinHeight)
        # self._hmax_edit.valueChanged.connect(self.worker.setMaxHeight)
        self.setWmin.connect(self.worker.setMinWidth)
        self.setWmax.connect(self.worker.setMaxWidth)
        self.setHmin.connect(self.worker.setMinHeight)
        self.setHmax.connect(self.worker.setMaxHeight)
        self.setRoi.connect(self.worker.setRoi)
        self.resetRoi.connect(self.worker.resetRoi)
        self._wdist.valueChanged.connect(self.worker.setDistThreshWatershed)
        self._intermediate_combo.currentTextChanged.connect(
            self.setIntermediateOutput)
        self.sigIntermediateOutput.connect(self.worker.setIntermediateOutput)
        self.sigProcess.connect(self.worker.process)
        self.sigSetOutlineStyle.connect(self.worker.setOutlineStyle)
        self.worker.sigProcessed.connect(self.sigProcessed)
        self.worker.sigSegPolygons.connect(self.sigSegPolygons)
        self.worker.sigIntermediate.connect(self._intermediate_win.setFrame)
        self.sigQuit.connect(self.saveSettings)
        ###################
        # Thread setup
        self.sigQuit.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def setOutlineStyle(self, style: str) -> None:
        self.sigSetOutlineStyle.emit(outline_dict[style])

    def fixBboxOutline(self) -> None:
        self.outlineCombo.setCurrentText('bbox')
        self.outlineCombo.setEnabled(False)
        # self.sigSetOutlineStyle.emit(outline_dict['bbox'])

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
        if text in segmethod_dict:
            self.sigSegMethod.emit(segmethod_dict[text])
        else:
            raise NotImplementedError(f'{text} method not available')

    @qc.pyqtSlot(str)
    def setIntermediateOutput(self, text: str) -> None:
        if segstep_dict[text] == consts.SegStep.final:
            self._intermediate_win.hide()
        else:
            self._intermediate_win.setWindowTitle(text)
            self._intermediate_win.show()
        self.sigIntermediateOutput.emit(segstep_dict[text])

    @qc.pyqtSlot()
    def saveSettings(self):
        """Save the worker parameters"""
        settings.setValue('segment/blur_width', self.worker.kernel_width)
        settings.setValue('segment/blur_sd', self.worker.kernel_sd)
        settings.setValue('segment/thresh_invert', self.worker.invert)
        settings.setValue('segment/thresh_max_intensity',
                          self.worker.max_intensity)
        settings.setValue('segment/thresh_baseline', self.worker.baseline)
        settings.setValue('segment/thresh_blocksize', self.worker.block_size)
        settings.setValue('segment/dbscan_minsamples',
                          self.worker.dbscan_min_samples)
        settings.setValue('segment/dbscan_eps', self.worker.dbscan_eps)
        settings.setValue('segment/min_pixels', self.worker.pmin)
        settings.setValue('segment/max_pixels', self.worker.pmax)
        settings.setValue('segment/watershed_distthresh',
                          self.worker.wdist_thresh)

    def loadConfig(self, config):
        """TODO implement loading of configuration from YAML/dict"""
        raise NotImplementedError('This method is yet to be implemented')
        if 'blur_width' in config:
            self._blur_width_edit.setValue(config['blur_width'])
        if 'blur_sd' in config:
            self._blur_sd_edit.setValue(config['blur_sd'])
        if 'thresh_method' in config:
            if config['thresh_method'] == 'gaussian':
                self._thresh_method.setCurrentText('Adapative Gaussian')
            else:
                self._thresh_method.setCurrentText('Adapative Mean')
