# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-05 11:10 PM

"""Classical image-processing-based segmentation"""

from typing import List, Tuple
import logging
from collections import OrderedDict
import numpy as np
import cv2
from sklearn import cluster
from PyQt5 import (
    QtWidgets as qw,
    QtCore as qc
)

import argos.constants
from argos import utility as ut
from argos.display import Display


settings = ut.init()


def segment_by_dbscan(binary_img: np.ndarray, eps: float=5,
                      min_samples: int=10) -> List[np.ndarray]:
    """Use DBSCAN clustering to segment binary image.

    Parameters
    ----------
    binary_img: np.ndarray
        binary image, a 2D array containing 0s and 1s (obtaind by thresholding
        original image converted to grayscale).
    eps: float
        the epsilon parameter of DBSCAN.
    min_samples: int
        minimum number of pixels each cluster (object) must contain in order to
        be considered a valid object.

    Returns
    -------
    list
        List of coordinate arrays where the n-th entry is the array of
        positions of the pixels belonging to the n-th segmented object.
    """
    indices = np.nonzero(binary_img)
    xy = np.vstack((indices[1], indices[0])).T
    core, labels = cluster.dbscan(xy, eps=eps, min_samples=min_samples,
                                  metric='euclidean', algorithm='auto')
    unique_labels = set(labels)
    unique_labels.discard(-1)  # -1 is the noise label
    return [xy[labels == label] for label in sorted(unique_labels)]


def segment_by_contours(binary_img: np.ndarray) -> List[np.ndarray]:
    """Segment binary image by finding contours of contiguous
    nonzero pixels and then filling those contours with an integer
    color value.

    Although, this is also part of the watershed algorithm, for small objects
    that are clearly separable from the background and each other, this
    works equally well.

    Parameters
    ----------
    binary_img: numpy.ndarray
        binary input image (obtained by thresholding grayscale image).

    Returns
    -------
    list
        List of coordinate arrays where the n-th entry is the array of
        positions of the pixels belonging to the n-th segmented object.
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
        points[:, 0], points[:, 1] = points[:, 1], points[:, 0].copy()
    return ret



def segment_by_watershed(binary_img: np.ndarray, img: np.ndarray,
                         dist_thresh: float=3.0) -> Tuple[np.ndarray,
                                                          List[np.ndarray]]:
    """Segment image using watershed algorithm.

    Parameters
    ----------
    binary_img:np.ndarray
        Binary image derived from ``img`` with nonzero pixel blobs for objects.
        This is usually produced after converting the ``img`` to grayscale and
        then thresholding.
    img: np.ndarray
        Original image to be segmented.
    dist_thresh: float, optional
        Threshold for distance of pixels from boundary to consider them core
        points. If it is < 1.0, it is interpreted as fraction of the maximum
        of the distances.

    Returns
    -------
    markers: np.ndarray[uint8]
        2D array with same as ``binary_img`` with a positive integer labelling
        the x, y coordinates for each segmented object.
    points_list: list[np.ndarray]
        List of arrays containing positions of the pixels in each object.

    Notes
    -----
    This code is derivative of this OpenCV tutorial:

    https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html

    and falls under the same licensing.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
    # Distance transform calculates the distance of each pixel from
    # background (black in this case) pixels. So we have an image
    # where pixel intensity means the distance of that pixel from the
    # background
    dist_xform = cv2.distanceTransform(opening, cv2.DIST_L2,
                                       cv2.DIST_MASK_PRECISE)
    # Thresholding the distance image to find pixels which are more
    # than a certain distance away from background - this should give
    # us the pixels central to foreground objects
    if dist_thresh < 1.0:
        # threshold relative to maximum of computed distance
        dist_thresh *= dist_xform.max()
    ret, sure_fg = cv2.threshold(dist_xform, dist_thresh, 255, 0)
    sure_fg = np.uint8(sure_fg)
    # border between background and foreground
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg, connectivity=4)
    logging.debug(f'Found {ret} connected components')
    # 0 is for background - assign a large value to keep them off
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    unique_labels = set(markers.flat)
    unique_labels.discard(-1)
    unique_labels.discard(0)
    ret = [np.argwhere(markers == label) for label in sorted(unique_labels)]
    markers[markers == -1] = 0
    markers = np.uint8(markers)
    # Fast swapping of y and x - see answer by blax here:
    # https://stackoverflow.com/questions/4857927/swapping-columns-in-a-numpy-array
    for points in ret:
        points[:, 0], points[:, 1] = points[:, 1], points[:, 0].copy()
    return markers, ret


def extract_valid(points_list, pmin, pmax, wmin, wmax, hmin, hmax):
    """
    Filter valid objects based on size limits.

    The length of the smaller side of the minimum bounding rotated-rectangle is
    considered width and the larger as height.

    Parameters
    ----------
    points_list: list[np.ndarray]
        List of coordinate arrays for pixels in each segmented object pixels.
    pmin: int
        Minimum number of pixels.
    pmax: int
        Maximum number of pixels.
    wmin: int
        Minimum width of minimum bounding rotated rectangle.
    wmax: int
        Maximum width of minimum bounding rotated rectangle.
    hmin: int
        Minimum height/length of minimum bounding rotated rectangle.
    hmax: int
        Maximum height/length of minimum bounding rotated rectangle.

    Returns
    -------
    list
        Coordinate arrays of objects that are between ``pmin`` and ``pmax``
        pixels, ``wmin`` and ``wmax`` width, and ``hmin`` and ``hmax`` height
        where The length of the smaller side of the minimum bounding
        rotated-rectangle is considered width and the larger as height.
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


def get_bounding_poly(points_list: List[np.ndarray],
                      style: ut.OutlineStyle) -> List[np.ndarray]:
    """This returns a list of bounding-polygons of the list of points
    in `points_list`.

    Parameters
    ----------
    points_list: list
        List of point arrays masking each object.
    style: argos.utility.OutlineStyle

    Returns
    -------
    list[np.ndarray]
        If `style` is OutlineStyle.fill - the same list of points without
        doing anything.
        If OutlineStyle.contour - the list of points representing the
        contour of each entry in `points_list`.
        If OutlineStyle.minrect - the list of vertices of the minimum-rotated
        rectangles bounding each entry in `points_list`.
        If OutlineStyle.bbox - the list of vertices of the axis-aligned
        rectangles bounding each entry in `points_list`.

    This does not strictly extract bounding points, as when `style` is
    `OutlineStyle.filled`, it just returns the same set of points. Any client
    using a uniform policy of drawing a polygon with the returned points will be
    essentially filling it up.

    I had to make a binary image with the specified points set to 1 because
    that's what cv2.findContours takes.
    """
    if style == ut.OutlineStyle.fill:
        return points_list
    contours_list = []
    for points in points_list:
        # logging.debug('%r, %r', type(points), points)
        if style == ut.OutlineStyle.minrect:
            contours_list.append(
                np.int0(cv2.boxPoints(cv2.minAreaRect(points))))
            continue
        rect = np.array(cv2.boundingRect(points))
        if style == ut.OutlineStyle.bbox:
            contours_list.append(ut.rect2points(rect))
        elif style == ut.OutlineStyle.contour:
            # Create a binary image with the size of the bounding box
            binary_img = np.zeros((rect[3], rect[2]), dtype=np.uint8)
            # Turn on the pixels for corresponding points
            pos = points - rect[:2]
            binary_img[pos[:, 1], pos[:, 0]] = 1
            contours, hierarchy = cv2.findContours(binary_img,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            # convert contour pixel positions back to image space
            contours = [contour.squeeze() + rect[:2] for contour in contours]
            contours_list += contours

    return contours_list


segstep_dict = OrderedDict([
    ('Final', argos.constants.SegStep.final),
    ('Blurred', argos.constants.SegStep.blur),
    ('Thresholded', argos.constants.SegStep.threshold),
    ('Segmented', argos.constants.SegStep.segmented),
    ('Filtered', argos.constants.SegStep.filtered),
    ])

segmethod_dict = OrderedDict([
    ('Threshold', argos.constants.SegmentationMethod.threshold),
    ('Watershed', argos.constants.SegmentationMethod.watershed),
    ('DBSCAN', argos.constants.SegmentationMethod.dbscan)
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
        self.baseline =10
        # segmentation method
        self.seg_method = argos.constants.SegmentationMethod.threshold
        #  DBSCAN parameters
        self.dbscan_eps = 5
        self.dbscan_min_samples =10
        # Watershed algorithm - distance threshold
        self.wdist_thresh = 3.0
        # cleanup params
        self.pmin = 10
        self.pmax = 500
        self.wmin = 20
        self.wmax = 50
        self.hmin = 50
        self.hmax = 200
        self.intermediate = argos.constants.SegStep.final
        self.cmap = cv2.COLORMAP_JET

    @qc.pyqtSlot(ut.OutlineStyle)
    def setOutlineStyle(self, mode: ut.OutlineStyle) -> None:
        self.outline_style = mode

    @qc.pyqtSlot(argos.constants.SegStep)
    def setIntermediateOutput(self, step: argos.constants.SegStep) -> None:
        self.intermediate = step

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

    @qc.pyqtSlot(int)
    def setBlockSize(self, value: int) -> None:
        if value % 2 == 0:
            self.block_size = value + 1
        else:
            self.block_size = value

    @qc.pyqtSlot(argos.constants.SegmentationMethod)
    def setSegmentationMethod(self, method: argos.constants.SegmentationMethod) -> None:
        self.seg_method = method

    @qc.pyqtSlot(float)
    def setEpsDBSCAN(self, value: int) -> None:
        self.dbscan_eps = value

    @qc.pyqtSlot(int)
    def setMinSamplesDBSCAN(self, value: int) -> None:
        self.dbscan_min_samples = value

    @qc.pyqtSlot(float)
    def setDistThreshWatershed(self, value: float) -> None:
        self.wdist_thresh = value

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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(
            gray,
            ksize=(self.kernel_width, self.kernel_width),
            sigmaX=self.kernel_sd)
        if self.intermediate == argos.constants.SegStep.blur:
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
        if self.intermediate == argos.constants.SegStep.threshold:
            self.sigIntermediate.emit(binary, pos)
        if self.seg_method == argos.constants.SegmentationMethod.threshold:
            seg = segment_by_contours(binary)
        elif self.seg_method == argos.constants.SegmentationMethod.dbscan:
            seg = segment_by_dbscan(binary, self.dbscan_eps,
                                    self.dbscan_min_samples)
        elif self.seg_method == argos.constants.SegmentationMethod.watershed:
            processed, seg = segment_by_watershed(binary, image,
                                                  self.wdist_thresh)
        if self.intermediate == argos.constants.SegStep.segmented:
            if self.seg_method == argos.constants.SegmentationMethod.watershed:
                self.sigIntermediate.emit(cv2.applyColorMap(processed, self.cmap), pos)
            else:
                for ii, points in enumerate(seg):
                    binary[points[:, 1], points[:, 0]] = ii + 1
                self.sigIntermediate.emit(
                    cv2.applyColorMap(binary, self.cmap),
                    pos)
        seg = extract_valid(seg, self.pmin, self.pmax, self.wmin, self.wmax,
                            self.hmin, self.hmax)
        if self.intermediate == argos.constants.SegStep.filtered:
            for ii, points in enumerate(seg):
                binary[points[:, 1], points[:, 0]] = ii + 1
            self.sigIntermediate.emit(
                cv2.applyColorMap(binary, self.cmap),
                pos)
        if self.outline_style == ut.OutlineStyle.bbox:
            bboxes = [cv2.boundingRect(points) for points in seg]
            self.sigProcessed.emit(np.array(bboxes), pos)
        elif self.outline_style == ut.OutlineStyle.contour:
            contours = get_bounding_poly(seg, ut.OutlineStyle.contour)
            bboxes = {ii: contour for ii, contour in enumerate(contours)}
            self.sigSegPolygons.emit(bboxes, pos)


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
    sigSegMethod = qc.pyqtSignal(argos.constants.SegmentationMethod)
    sigIntermediateOutput = qc.pyqtSignal(argos.constants.SegStep)

    sigQuit = qc.pyqtSignal()


    def __init__(self, *args, **kwargs):
        super(SegWidget, self).__init__(*args, **kwargs)
        self.worker = SegWorker()
        layout = qw.QFormLayout()
        self._blur_width_label = qw.QLabel('Blur width')
        self._blur_width_edit = qw.QSpinBox()
        self._blur_width_edit.setRange(1, 100)
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
        self._dbscan_minsamples.setRange(1, 1000)
        value = settings.value('segment/dbscan_minsamples',
                               self.worker.dbscan_min_samples,
                               type=int)
        self._dbscan_minsamples.setValue(value)
        self.worker.dbscan_min_samples = value
        layout.addRow(self._dbscan_minsamples_label, self._dbscan_minsamples)
        self._dbscan_eps_label = qw.QLabel('DBSCAN epsilon')
        self._dbscan_eps = qw.QDoubleSpinBox()
        self._dbscan_eps.setRange(0.1, 100)
        self._dbscan_eps.setStepType(qw.QAbstractSpinBox.AdaptiveDecimalStepType)
        value = settings.value('segment/dbscan_eps',
                               self.worker.dbscan_eps,
                               type=float)
        self._dbscan_eps.setValue(value)
        self.worker.dbscan_eps = value
        layout.addRow(self._dbscan_eps_label, self._dbscan_eps)
        self._pmin_label = qw.QLabel('Minimum pixels')
        self._pmin_edit = qw.QSpinBox()
        self._pmin_edit.setRange(1, 1000)
        value = settings.value('segment/min_pixels',
                               self.worker.pmin,
                               type=int)
        self._pmin_edit.setValue(value)
        self.worker.pmin = value
        layout.addRow(self._pmin_label, self._pmin_edit)
        self._pmax_label = qw.QLabel('Maximum pixels')
        self._pmax_edit = qw.QSpinBox()
        self._pmax_edit.setRange(1, 1000)
        value = settings.value('segment/max_pixels',
                               self.worker.pmax,
                               type=int)
        self._pmax_edit.setValue(value)
        self.worker.pmax = value
        layout.addRow(self._pmax_label, self._pmax_edit)
        self._wmin_label = qw.QLabel('Minimum width')
        self._wmin_edit = qw.QSpinBox()
        self._wmin_edit.setRange(1, 1000)
        value = settings.value('segment/min_width',
                               self.worker.wmin,
                               type=int)
        self._wmin_edit.setValue(value)
        self.worker.wmin = value
        layout.addRow(self._wmin_label, self._wmin_edit)
        self._wmax_label = qw.QLabel('Maximum width')
        self._wmax_edit = qw.QSpinBox()
        self._wmax_edit.setRange(1, 1000)
        value = settings.value('segment/max_width',
                               self.worker.wmax,
                               type=int)
        self._wmax_edit.setValue(value)
        self.worker.wmax = value
        layout.addRow(self._wmax_label, self._wmax_edit)
        self._hmin_label = qw.QLabel('Minimum length')
        self._hmin_edit = qw.QSpinBox()
        self._hmin_edit.setRange(1, 1000)
        value = settings.value('segment/min_height',
                               self.worker.hmin,
                               type=int)
        self._hmin_edit.setValue(value)
        self.worker.hmin = value
        layout.addRow(self._hmin_label, self._hmin_edit)
        self._hmax_label = qw.QLabel('Maximum length')
        self._hmax_edit = qw.QSpinBox()
        self._hmax_edit.setRange(1, 1000)
        value = settings.value('segment/max_height',
                               self.worker.hmax,
                               type=int)
        self._hmax_edit.setValue(value)
        self.worker.hmax = value
        layout.addRow(self._hmax_label, self._hmax_edit)
        self._wdist_label = qw.QLabel('Distance threshold')
        self._wdist = qw.QDoubleSpinBox()
        self._wdist.setRange(0, 10)
        self._wdist.setStepType(qw.QAbstractSpinBox.AdaptiveDecimalStepType)
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
        self.setLayout(layout)

        self._intermediate_win = Display()
        self._intermediate_win.setWindowFlag(qc.Qt.Window + qc.Qt.WindowStaysOnTopHint)
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
        self._wmin_edit.valueChanged.connect(self.worker.setMinWidth)
        self._wmax_edit.valueChanged.connect(self.worker.setMaxWidth)
        self._hmin_edit.valueChanged.connect(self.worker.setMinHeight)
        self._hmax_edit.valueChanged.connect(self.worker.setMaxHeight)
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

    def setOutlineStyle(self, style: ut.OutlineStyle) -> None:
        self.sigSetOutlineStyle.emit(style)

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
        if segstep_dict[text] == argos.constants.SegStep.final:
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
        settings.setValue('segment/min_width', self.worker.wmin)
        settings.setValue('segment/max_width', self.worker.wmax)
        settings.setValue('segment/min_height', self.worker.hmin)
        settings.setValue('segment/max_height', self.worker.hmax)
        settings.setValue('segment/watershed_distthresh',
                          self.worker.wdist_thresh)

