# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-07-14 1:39 PM
"""
===============================================
Track objects in batch mode (non-interactively)
===============================================
Usage:
::
     python -m argos_track.batchtrack -i {input_file} -o {output_file}
     -c {config_file}

Try ``python -m argos_track.batchtrack -h`` for details of command-line
options.

This program allows non-interactive tracking of objects in a video.
When using classical segmentation this can speed things up by
utilizing multiple CPU cores.

It may be easier to use the interactive tracking :py:mod:`argos_track`
to play with the segmentation parameters to see what work best for
videos in a specific setting. The optimal setting can then be exported
to a configuration file which will then be passed with ``-c`` command
line option .

Examples
--------
Use YOLACT for segmentation and SORT for tracking:
::
    python -m argos_track.batchtrack -i video.avi -o video.h5 -m yolact \\
    --yconfig=config/yolact.yml -w config/weights.pth -s 0.1 -k 10 \\
    --overlap_thresh=0.3 --cuda \\
    --pmin=10 --pmax=500 --wmin=5 --wmax=100 --hmin=5 --hmax=100 \\
    -x 0.3 --min_hits=3 --max_age=20

The above command tells the ``batchtrack`` script to read the input
video ``video.avi`` and write the output to the file ``video.h5``. The
rest of the arguments:

- ``-m yolact`` tells it to use YOLACT as the segmentation method.

- ``--yconfig=config/yolact.yml``: Read YOLACT settings from the file
``config/yolact.yml``

- ``-w config/weights.pth``: Read YOLACT neural network weights from
the file ``config/weights.pth``.

- ``-s 0.1``: Include detections with score above 0.1

- ``-k 10``: Keep only the top 10 detections.

- ``--overlap_thresh=0.3``: At segmentation stage, merge detections
  whose bounding boxes overlap more than 0.3 of their total area.

- ``--cuda``: use GPU acceleration.

- ``--pmin=10``: Include objects at least 10 pixels in bounding box
  area.

- ``--pmax=500``: Include objects at most 500 pixels in bounding box
  area.

- ``--wmin=5``: Include objects at least 5 pixels wide.

- ``--wmax=100``: Include objects at most 100 pixels wide.

- ``--hmin=5``: Include objects at least 5 pixels long.

- ``--hmax=100``: Include objects at most 100 pixels long.

- ``-x 0.3``: In the tracking stage, if objects in two successive
  frames overlap more than 0.3 times their combined area, then
  consider them to be the same object.

- ``--min_hits=3``: An object must be detcted at least in 3
  consecutive frames to be included in the tracks.

- ``--max_age=20``: If an object cannot be matched to any detected
  object across 20 successive frames, then discard it (possibly it
  exited the view). [Remember that if you have a 30 frames per second
  video, 20 frames means 2/3 second in real time.]


All of this can be more easily set graphically in
:py:mod:`argos_track` and exported into a file, which can then be
passed with ``-c {config_file}``.

"""
import argparse
import os
from collections import namedtuple
import sys
import logging
from functools import partial, wraps
import numpy as np
import cv2
import yaml
import time
from datetime import timedelta
import pandas as pd
import concurrent.futures as cf
import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from yolact import Yolact
from yolact.data import config as yconfig

# This is actually yolact.utils
from yolact.utils.augmentations import FastBaseTransform
from yolact.layers import output_utils as oututils
from argos_track.sortracker import SORTracker
from argos.constants import DistanceMetric, OutlineStyle
import argos.utility as ut
from argos.segment import (
    segment_by_contour_bbox,
    segment_by_dbscan,
    segment_by_watershed,
    extract_valid,
)

# NOTE: defaults in namedtuple allowed only in Python3.7+
ThreshParam = namedtuple(
    'ThreshParam',
    (
        'blur_width',
        'blur_sd',
        'invert',
        'method',
        'max_intensity',
        'baseline',
        'blocksize',
    ),
    defaults=(7, 1, True, 'gaussian', 255, 10, 41),
)

LimitParam = namedtuple(
    'LimitParam',
    ('pmin', 'pmax', 'wmin', 'wmax', 'hmin', 'hmax'),
    defaults=(10, 500, 10, 50, 10, 200),
)

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s '
    'p=%(processName)s[%(process)d] '
    't=%(threadName)s[%(thread)d] '
    '%(filename)s#%(lineno)d:%(funcName)s: '
    '%(message)s',
    level=logging.INFO,
)

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def timed(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        tic = time.perf_counter_ns()
        try:
            return func(*args, **kwargs)
        finally:
            toc = time.perf_counter_ns()
            dt = (toc - tic) / 1e9
            dt = timedelta(seconds=dt)
            msg = f'{func}: Total execution time: {dt}'
            print(msg)
            logging.info(msg)

    return _time_it


# Global yolact network weights
ynet = None
config = None


def load_config(filename):
    """Load YOLACT configuration."""
    global config
    config = yconfig.cfg
    if filename == '':
        return
    print(f'Loading config from {filename}')
    with open(filename, 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
        for key, value in cfg.items():
            config.__setattr__(key, value)
        if 'mask_proto_debug' not in cfg:
            config.mask_proto_debug = False


def load_weights(filename, cuda):
    """Load YOLACT network weights"""
    global ynet
    if filename == '':
        raise ValueError('Empty filename for network weights')
    print('#### CUDA ENABLED', cuda)
    print(f'Loading weights from {filename}')
    tic = time.perf_counter_ns()
    with torch.no_grad():
        if cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        # torch.set_default_tensor_type('torch.FloatTensor')
        ynet = Yolact()
        ynet.load_weights(filename, False)
        ynet.eval()
    toc = time.perf_counter_ns()
    logging.debug(f'Time to load weights: {1e-9 * (toc - tic)}')


@timed
def init_yolact(cfgfile, netfile, cuda):
    """Initialize YOLACT"""
    load_config(cfgfile)
    load_weights(netfile, cuda)


# This function should stay here for it uses the globals
# @timed
def segment_yolact(
    frame, score_threshold, top_k, overlap_thresh, cfgfile, netfile, cuda
):
    """Segment objects in frame using YOLACT.

    Parameters
    ----------
    frame: numpy.ndarray
        (WxHxC) integer array with the image content.
    score_threshold: float
        Minimum score to include object, should be in `(0, 1)`.
    top_k: int
        The number of segmented objects to keep.
    overlap_thresh: float
        Merge objects whose bounding boxes overlap (intersection over union)
        more than this amount.
    cfgfile: str
        Path to YOLACT configuration file.
    netfile: str
        Path to YOLACT network weights file.
    cuda: bool
        Whether to use CUDA.
    Returns
    -------
    numpy.ndarray
        An array of bounding boxes of detected objects in
        (xleft, ytop, width, height) format.
    """
    global ynet
    global config

    if ynet is None:
        init_yolact(cfgfile, netfile, cuda)
    # Partly follows yolact eval.py
    tic = time.perf_counter_ns()
    with torch.no_grad():
        if cuda:
            frame = torch.from_numpy(frame).cuda().float()
        else:
            frame = torch.from_numpy(frame).float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = ynet(batch)
        h, w, _ = frame.shape
        config.rescore_bbox = True
        classes, scores, boxes, masks = oututils.postprocess(
            preds,
            w,
            h,
            visualize_lincomb=False,
            crop_masks=True,
            score_threshold=score_threshold,
        )
        idx = scores.argsort(0, descending=True)[:top_k]
        # if self.config.eval_mask_branch:
        #     masks = masks[idx]
        classes, scores, boxes = [
            x[idx].cpu().numpy() for x in (classes, scores, boxes)
        ]
        # This is probably not required, `postprocess` uses
        # `score_thresh` already
        # num_dets_to_consider = min(self.top_k, classes.shape[0])
        # for j in range(num_dets_to_consider):
        #     if scores[j] < self.score_threshold:
        #         num_dets_to_consider = j
        #         break
        # logging.debug('Bounding boxes: %r', boxes)
        # Convert from top-left bottom-right format to
        # top-left, width, height format
        if len(boxes) == 0:
            return np.empty(0)

        boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        boxes = np.asanyarray(np.rint(boxes), dtype=np.int_)
        if overlap_thresh < 1:
            dist_matrix = ut.pairwise_distance(
                new_bboxes=boxes,
                bboxes=boxes,
                boxtype=OutlineStyle.bbox,
                metric=DistanceMetric.iou,
            )
            bad_boxes = []
            for ii in range(dist_matrix.shape[0] - 1):
                for jj in range(ii + 1, dist_matrix.shape[1]):
                    if dist_matrix[ii, jj] < 1 - overlap_thresh:
                        bad_boxes.append(jj)
            boxes = np.array(
                [
                    boxes[ii]
                    for ii in range(boxes.shape[0])
                    if ii not in bad_boxes
                ],
                dtype=np.int_,
            )
        toc = time.perf_counter_ns()
        logging.debug('Time to process single image: %f s', 1e-9 * (toc - tic))
        return boxes


def read_frame(video):
    """Read a frame from `video` and return the frame number and the image data"""
    if (video is None) or not video.isOpened():
        return (-1, None)
    frame_no = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = video.read()
    return (frame_no, frame)


def threshold(frame: np.ndarray, params: ThreshParam):
    """Adaptive threshold image after converting to grayscale and blurring.

    Parameters
    ----------
    frame: numpy.ndarray
        Image data.
    params: ThreshParam
        Parameters for thresholding.
    Returns
    -------
    np.ndarray
        Binary image from thresholding.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(
        gray,
        ksize=(params.blur_width, params.blur_width),
        sigmaX=params.blur_sd,
    )
    if params.invert:
        thresh_type = cv2.THRESH_BINARY_INV
    else:
        thresh_type = cv2.THRESH_BINARY
    if params.method == 'gaussian':
        thresh_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    elif params.method == 'mean':
        thresh_method = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        raise ValueError(f'Invalid thresholding method {params.method}')
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=params.max_intensity,
        adaptiveMethod=thresh_method,
        thresholdType=thresh_type,
        blockSize=params.blocksize,
        C=params.baseline,
    )
    return binary


def bbox_func(points_list):
    """Calculate list of bounding boxes of point arrays"""
    ret = np.array([cv2.boundingRect(points) for points in points_list])
    return ret


def create_seg_func_list(args):
    """Create a sequence of functions to be applied on image for
    segmentation using classical methods.

    Classical segmentation involves blurring and thresholding followed
    by additional procedures depending on the specific method.

    Parameters
    ----------
    args: dict
        Argument dictionary.
    Returns
    -------
    list
        List of functions to be applied in sequence on the input image.
    """
    thresh_params = ThreshParam(
        blur_width=args.blur_width,
        blur_sd=args.blur_sd,
        invert=args.thresh_invert,
        method=args.thresh_method,
        max_intensity=args.thresh_max,
        baseline=args.thresh_baseline,
        blocksize=args.thresh_blocksize,
    )
    thresh_func = partial(threshold, params=thresh_params)

    seg_method = args.seg_method
    if seg_method == 'threshold':
        seg_func = segment_by_contour_bbox
    elif seg_method == 'watershed':
        seg_func = partial(segment_by_watershed, dist_thresh=args.dist_thresh)
    elif seg_method == 'dbscan':
        seg_func = partial(
            segment_by_dbscan, eps=args.eps, min_samples=args.min_samples
        )
    else:
        raise ValueError(f'Unknown segmentation method: {seg_method}')

    limit_func = partial(
        extract_valid,
        pmin=args.pmin,
        pmax=args.pmax,
        wmin=args.wmin,
        wmax=args.wmax,
        hmin=args.hmin,
        hmax=args.hmax,
    )
    return [thresh_func, seg_func, limit_func, bbox_func]


# @timed
def run_fn_seq(fn_args):
    """Run frame through a function pipeline.

    Parameters
    ----------
    fn_args: tuple
        tuple of the form ((f0, a0), (f1, a1), ...) where `f0`, `f1`, etc. are
         functions and `a0`, `a1`, etc. are the arguments (tuple) of the
         corresponding function.
    Could not use `reduce` as the number of arguments varies depending  on the
    function.
    """
    result = None
    for fn, args in fn_args:
        if result is None:
            result = fn(*args)
        else:
            result = fn(result, *args)
    return result


@timed
def batch_segment(args):
    """Segment frames in parallel and save the bboxes of segmented objects in
    an HDF file for later tracking"""
    if (
        'SLURM_JOB_ID' in os.environ
    ):  # this is a slurm job, use slurm env var or safe number of 2
        cpu_count = int(os.environ.get('SLURM_CPUS_PER_TASK', '2'))
    else:
        cpu_count = mp.cpu_count()
    max_workers = args.max_proc if args.max_proc > 0 else cpu_count
    print('*** Using cuda:', args.cuda)
    if args.seg_method == 'yolact':
        seg_fn = partial(
            segment_yolact,
            score_threshold=args.score_thresh,
            top_k=args.top_k,
            overlap_thresh=args.overlap_thresh,
            cfgfile=args.yconfig,
            netfile=args.weight,
            cuda=args.cuda,
        )
        max_workers = max(1, min(cpu_count, torch.cuda.device_count()))
    else:
        thresh_fn, seg_fn, limit_fn, bbox_fn = create_seg_func_list(args)
    cv2.setNumThreads(cpu_count)
    torch.set_num_threads(cpu_count)
    print('Workers:', max_workers, 'CPUs:', cpu_count)
    logging.info(
        f'Running segmentation with {max_workers} worker processes on system with {cpu_count} cpus.'
    )
    video = cv2.VideoCapture(args.infile)
    if not video.isOpened():
        raise IOError('Could not open video')
    data = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        while video.isOpened():
            frames = []
            for ii in range(max_workers):
                frame_no, frame = read_frame(video)
                if frame_no % 100 == 0:
                    logging.debug(f'Read till {frame_no}')
                if frame is None:
                    video.release()
                    break
                frames.append((frame_no, frame))
            futures = {}
            for frame_no, frame in frames:
                if args.seg_method == 'yolact':
                    fut = executor.submit(seg_fn, frame)
                else:
                    seg_arg = (
                        (frame,) if args.seg_method == 'watershed' else ()
                    )
                    fut = executor.submit(
                        run_fn_seq,
                        (
                            (thresh_fn, (frame,)),
                            (seg_fn, seg_arg),
                            (limit_fn, ()),
                            (bbox_fn, ()),
                        ),
                    )
                futures[fut] = frame_no
            while futures:
                done, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    frame_no = futures.pop(fut)
                    result = fut.result()
                    for ii, bbox in enumerate(result):
                        if len(bbox) > 0:
                            data.append(
                                {
                                    'frame': frame_no,
                                    'x': bbox[0],
                                    'y': bbox[1],
                                    'w': bbox[2],
                                    'h': bbox[3],
                                }
                            )
    if len(data) == 0:
        raise RuntimeError('Data list empty')
    data = pd.DataFrame(data)
    data.sort_values(by='frame', inplace=True)
    data.to_hdf(args.outfile, 'segmented')
    logging.info(f'Data saved in {args.outfile} under /segmented')


@timed
def batch_track(args):
    """outfile should have a `/segmented` table containing the segmentation data.

    saves the result in same file under `/tracked`
    """
    segments = pd.read_hdf(args.outfile, 'segmented')
    results = []
    if args.sort_metric == 'iou':
        metric = DistanceMetric.iou
    else:
        metric = DistanceMetric.euclidean
    tracker = SORTracker(
        metric=metric,
        min_dist=args.min_dist,
        max_age=args.max_age,
        n_init=args.min_hits,
        min_hits=args.min_hits,
    )
    for frame, fgrp in segments.groupby('frame'):
        if len(fgrp) == 0:
            continue
        tracked = tracker.update(
            fgrp[['x', 'y', 'w', 'h']].astype(np.int_).values
        )
        for tid, bbox in tracked.items():
            results.append(
                {
                    'frame': frame,
                    'trackid': tid,
                    'x': bbox[0],
                    'y': bbox[1],
                    'w': bbox[2],
                    'h': bbox[3],
                }
            )

        if frame % 100 == 0:
            logging.info(f'Processed till {frame}')
    results = pd.DataFrame(results)
    results.sort_values(by='frame', inplace=True)
    results.to_hdf(args.outfile, 'tracked')
    logging.info(f'Tracking data saved in {args.outfile} under /tracked.')


def make_parser():
    parser = argparse.ArgumentParser('Track objects in video in batch mode')
    parser.add_argument('-i', '--infile', type=str, help='input file')
    parser.add_argument(
        '-o',
        '--outfile',
        type=str,
        help='output file. Create an HDF file with segmentation'
        ' in `segmented` and tracking data in the table'
        ' `tracked`',
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='configuration file to use for rest of the' ' arguments',
    )
    parser.add_argument(
        '-p',
        '--max_proc',
        type=int,
        default=-1,
        help='number of parallel processes',
    )
    parser.add_argument(
        '-m',
        '--seg_method',
        type=str,
        default='yolact',
        help='method for segmentation' ' (yolact/threshold/watershed/dbscan)',
    )
    yolact_grp = parser.add_argument_group(
        'YOLACT', 'Parameters for YOLACT segmentation'
    )
    yolact_grp.add_argument(
        '--yconfig', type=str, help='YOLACT configuration file'
    )
    yolact_grp.add_argument(
        '-w', '--weight', type=str, help='YOLACT trained weights file'
    )
    yolact_grp.add_argument(
        '-s',
        '--score_thresh',
        type=float,
        default=0.3,
        help='score threshold for accepting a detected object',
    )
    yolact_grp.add_argument(
        '-k',
        '--top_k',
        type=int,
        default=30,
        help='maximum number of objects above score' ' threshold to keep',
    )
    yolact_grp.add_argument(
        '--overlap_thresh',
        type=float,
        default=1.0,
        help='Bboxes with IoU overlap higher than this are' ' merged',
    )
    yolact_grp.add_argument(
        '--cuda', action='store_true', help='If specified, use CUDA'
    )
    thresh_grp = parser.add_argument_group(
        'Threshold', 'Parameters for thresholding'
    )
    thresh_grp.add_argument(
        '--thresh_method',
        type=str,
        default='gaussian',
        help='Method for adaptive thresholding' ' (gaussian/mean)',
    )
    thresh_grp.add_argument(
        '--thresh_max',
        type=int,
        default=255,
        help='Maximum intensity for thresholding',
    )
    thresh_grp.add_argument(
        '--thresh_baseline',
        type=int,
        default=10,
        help='baseline intensity for thresholding',
    )
    thresh_grp.add_argument(
        '--blur_width',
        type=int,
        default=7,
        help='blur width before thresholding.' ' Must be odd number.',
    )
    thresh_grp.add_argument(
        '--blur_sd',
        type=float,
        default=1.0,
        help='SD for Gaussian blur before thresholding',
    )
    thresh_grp.add_argument(
        '--thresh_blocksize',
        type=int,
        default=41,
        help='block size for adaptive thresholding.' ' Must be odd number',
    )
    thresh_grp.add_argument(
        '--thresh_invert',
        type=bool,
        default=True,
        help='Inverted thresholding',
    )
    watershed_grp = parser.add_argument_group(
        'Watershed', 'Parameter for segmentation using watershed algorithm'
    )
    watershed_grp.add_argument(
        '-d',
        '--dist_thresh',
        type=float,
        default=3.0,
        help='minimum distance of pixels from detected '
        'boundary to consider them core points',
    )
    dbscan_grp = parser.add_argument_group(
        'DBSCAN',
        'Parameters for segmentation by clustering pixels with'
        ' DBSCAN algorithm',
    )
    dbscan_grp.add_argument(
        '-e',
        '--eps',
        type=float,
        default=5.0,
        help='epsilon parameter for DBSCAN',
    )
    dbscan_grp.add_argument(
        '--min_samples',
        type=int,
        default=10,
        help='minimum number of pixels in each cluster' ' for DBSCAN',
    )
    limits_grp = parser.add_argument_group(
        'Limits', 'Parameters to set limits on detected object size.'
    )
    limits_grp.add_argument(
        '--pmin', type=int, default=10, help='Minimum number of pixels'
    )
    limits_grp.add_argument(
        '--pmax', type=int, default=500, help='Maximum number of pixels'
    )
    limits_grp.add_argument(
        '--hmin',
        type=int,
        default=10,
        help='Minimum height (longer side) of bounding box' ' in pixels',
    )
    limits_grp.add_argument(
        '--hmax',
        type=int,
        default=200,
        help='Maximum height (longer side) of bounding box' ' in pixels',
    )
    limits_grp.add_argument(
        '--wmin',
        type=int,
        default=10,
        help='Minimum width (shorter side) of bounding box' ' in pixels',
    )
    limits_grp.add_argument(
        '--wmax',
        type=int,
        default=100,
        help='Maximum width (shorter side) of bounding box' ' in pixels',
    )
    track_grp = parser.add_argument_group(
        'Tracker', 'Parameters for SORT tracker'
    )
    track_grp.add_argument(
        '--sort_metric',
        type=str,
        default='iou',
        help='Metric for measuring closeness.' ' iou or euclidean',
    )
    track_grp.add_argument(
        '-x',
        '--min_dist',
        type=float,
        default=0.3,
        help='Minimum distance between bounding boxes.'
        ' If iou, this their intersection as a'
        ' fraction of their combined area.'
        ' If euclidean, number of pixels.',
    )
    track_grp.add_argument(
        '--min_hits',
        type=int,
        default=3,
        help='Minimum number of hits to accept a track',
    )
    track_grp.add_argument(
        '--max_age',
        type=int,
        default=50,
        help='Maximum number of misses to exclude a track',
    )
    parser.add_argument(
        '--debug', action='store_true', help='Print debug info'
    )
    return parser


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # 2 proc 40 / 124 fps
    # 5 proc 25 / 50 fps
    parser = make_parser()
    args = parser.parse_args()
    print('ARGS:')
    print(args)
    if args.config is not None:
        with open(args.config, 'r') as cfg_file:
            config = yaml.safe_load(cfg_file)
            for key, value in config.items():
                vars(args)[key] = value
        print('ARGS after update from config:')
        print(args)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    batch_segment(args)
    batch_track(args)
    print('Finished tracking')
