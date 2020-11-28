# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-07-14 1:39 PM
"""Batch processing utility for tracking object in video.

This works using multiple processes to utilize multiple CPU cores.
"""
import argparse
from collections import namedtuple
import sys
import logging
from functools import partial, wraps
import numpy as np
import cv2
import yaml
import time
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
from argos.sortracker import KalmanTracker
from argos.constants import DistanceMetric, OutlineStyle
import argos.utility as ut
from argos.segment import (
    segment_by_contours,
    segment_by_dbscan,
    segment_by_watershed,
    extract_valid
)

# NOTE: defaults in namedtuple allowed only in Python3.7+
ThreshParam = namedtuple('ThreshParam', ('blur_width',
                                         'blur_sd',
                                         'invert',
                                         'method',
                                         'max_intensity',
                                         'baseline',
                                         'blocksize'),
                         defaults=(7, 1, True, 'gaussian', 255, 10, 41))

LimitParam = namedtuple('LimitParam', ('pmin',
                                       'pmax',
                                       'wmin',
                                       'wmax',
                                       'hmin',
                                       'hmax'),
                        defaults=(10, 500, 10, 50, 10, 200))

logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s '
                           'p=%(processName)s[%(process)d] '
                           't=%(threadName)s[%(thread)d] '
                           '%(filename)s#%(lineno)d:%(funcName)s: '
                           '%(message)s',
                    level=logging.INFO)

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
            print(f'Total execution time {(toc - tic) / 1e9} s')

    return _time_it


# Global yolact network weights
ynet = None
config = None


def load_config(filename):
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
    global ynet
    if filename == '':
        raise ValueError('Empty filename for network weights')
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
    load_config(cfgfile)
    load_weights(netfile, cuda)


# This function should stay here for it uses the globals
# @timed
def segment_yolact(frame, score_threshold, top_k, cfgfile, netfile, cuda):
    """:returns (classes, scores, boxes)

    where `boxes` is an array of bounding boxes of detected objects in
    (xleft, ytop, width, height) format.

    `classes` is the class ids of the corresponding objects.

    `scores` are the computed class scores corresponding to the detected objects.
    Roughly high score indicates strong belief that the object belongs to
    the identified class.
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
        frame_gpu = frame / 255.0
        h, w, _ = frame.shape
        save = config.rescore_bbox
        config.rescore_bbox = True
        classes, scores, boxes, masks = oututils.postprocess(
            preds, w, h,
            visualize_lincomb=False,
            crop_masks=True,
            score_threshold=score_threshold)
        idx = scores.argsort(0, descending=True)[:top_k]
        # if self.config.eval_mask_branch:
        #     masks = masks[idx]
        classes, scores, boxes = [x[idx].cpu().numpy()
                                  for x in (classes, scores, boxes)]
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
        if len(boxes) > 0:
            boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        toc = time.perf_counter_ns()
        logging.debug('Time to process single image: %f s',
                      1e-9 * (toc - tic))
        return boxes


class SORTracker(object):
    """SORT algorithm implementation. This is same as SORTracker
    sortracker.py except avoids Qt and multi-threading complications.

    NOTE: accepts bounding boxes in (x, y, w, h) format.

    """

    def __init__(self, metric=DistanceMetric.iou, min_dist=0.3, max_age=1,
                 n_init=3, min_hits=3, boxtype=OutlineStyle.bbox):
        super(SORTracker, self).__init__()
        self.n_init = n_init
        self.min_hits = min_hits
        self.boxtype = boxtype
        self.metric = metric
        if self.metric == DistanceMetric.iou:
            self.min_dist = 1 - min_dist
        else:
            self.min_dist = min_dist
        self.max_age = max_age
        self.trackers = {}
        self._next_id = 1
        self.frame_count = 0

    def reset(self):
        logging.debug('Resetting trackers.')
        self.trackers = {}
        self._next_id = 1
        self.frame_count = 0

    def update(self, bboxes: np.ndarray):
        predicted_bboxes = {}
        for id_, tracker in self.trackers.items():
            prior = tracker.predict()
            if np.any(np.isnan(prior)) or np.any(
                    prior[:KalmanTracker.NDIM] < 0):
                logging.info(f'Found nan or negative in prior of {id_}')
                continue
            predicted_bboxes[id_] = prior[:KalmanTracker.NDIM]
        self.trackers = {id_: self.trackers[id_] for id_ in predicted_bboxes}
        for id_, bbox in predicted_bboxes.items():
            if np.any(bbox < 0):
                logging.debug(f'EEEE prediced bbox negative: {id_}: {bbox}')
        matched, new_unmatched, old_unmatched = ut.match_bboxes(
            predicted_bboxes,
            bboxes[:, :KalmanTracker.NDIM],
            boxtype=self.boxtype,
            metric=self.metric,
            max_dist=self.min_dist)
        for track_id, bbox_id in matched.items():
            self.trackers[track_id].update(bboxes[bbox_id])
        for ii in new_unmatched:
            self._add_tracker(bboxes[ii, :KalmanTracker.NDIM])
        ret = {}
        for id_ in list(self.trackers.keys()):
            tracker = self.trackers[id_]
            if (tracker.time_since_update < 1) and \
                    (tracker.hits >= self.min_hits or
                     self.frame_count <= self.min_hits):
                ret[id_] = tracker.pos
            if tracker.time_since_update > self.max_age:
                self.trackers.pop(id_)
        return ret

    def _add_tracker(self, bbox):
        self.trackers[self._next_id] = KalmanTracker(bbox, self._next_id,
                                                     self.n_init,
                                                     self.max_age)
        self._next_id += 1


def read_frame(video):
    """Read a frame from `video` and return the frame number and the image data"""
    if (video is None) or not video.isOpened():
        return (None, -1)

    frame_no = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = video.read()
    logging.debug('Read frame no %d', frame_no)
    return (frame_no, frame)


def threshold(frame: np.ndarray, params: ThreshParam):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, ksize=(params.blur_width, params.blur_width),
                            sigmaX=params.blur_sd)
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
    binary = cv2.adaptiveThreshold(gray,
                                   maxValue=params.max_intensity,
                                   adaptiveMethod=thresh_method,
                                   thresholdType=thresh_type,
                                   blockSize=params.blocksize,
                                   C=params.baseline)
    return binary


def create_seg_func_list(args):
    thresh_params = ThreshParam(blur_width=args.blur_width,
                                blur_sd=args.blur_sd,
                                invert=args.invert_thresh,
                                method=args.thresh_method,
                                max_intensity=args.thresh_max,
                                baseline=args.thresh_baseline,
                                blocksize=args.thresh_blocksize)
    thresh_func = partial(threshold, params=thresh_params)

    seg_method = args.seg_method
    if seg_method == 'threshold':
        seg_func = segment_by_contours
    elif seg_method == 'watershed':
        seg_func = partial(segment_by_watershed, dist_thresh=args.dist_thresh)
    elif seg_method == 'dbscan':
        seg_func = partial(segment_by_dbscan, eps=args.eps,
                           min_samples=args.min_samples)
    else:
        raise ValueError(f'Unknown segmentation method: {seg_method}')

    limits_params = LimitParam(pmin=args.pmin,
                               pmax=args.pmax,
                               wmin=args.wmin,
                               wmax=args.wmax,
                               hmin=args.hmin,
                               hmax=args.hmax)
    limit_func = partial(extract_valid, params=limits_params)
    bbox_func = lambda points_list: [cv2.boundingRect(points)
                                     for points in points_list]
    return [thresh_func, seg_func, limit_func, bbox_func]


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
    cpu_count = mp.cpu_count()
    max_workers = cpu_count
    if args.seg_method == 'yolact':
        seg_fn = partial(segment_yolact, score_threshold=args.score,
                         top_k=args.top_k,
                         cfgfile=args.yconfig, netfile=args.weight,
                         cuda=args.cuda)
        max_workers = max(1, min(cpu_count, torch.cuda.device_count()))
    else:
        thresh_fn, seg_fn, limit_fn, bbox_fn = create_seg_func_list(args)
    logging.debug(f'Running segmentation with {max_workers} worker processes')
    video = cv2.VideoCapture(args.infile)
    data = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        while video.isOpened():
            frames = []
            for ii in range(cpu_count):
                frame_no, frame = read_frame(video)
                if frame is None:
                    break
                frames.append((frame_no, frame))
            futures = {}
            for frame_no, frame in frames:
                if args.seg_method == 'yolact':
                    fut = executor.submit(seg_fn, frame)
                else:
                    seg_arg = (frame,) if args.seg_method == 'watershed' else ()
                    fut = executor.submit(run_fn_seq, ((thresh_fn, (frame,)),
                                                       (seg_fn, seg_arg),
                                                       (limit_fn, ()),
                                                       (bbox_fn, ())))
                futures[fut] = frame_no
            while futures:
                done, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    frame_no = futures.pop(fut)
                    result = fut.result()
                    for ii, bbox in enumerate(result):
                        data.append({'frame': frame_no,
                                     'x': bbox[0],
                                     'y': bbox[1],
                                     'w': bbox[2],
                                     'h': bbox[3]})
    if len(data) == 0:
        raise RuntimeError('Data list empty')
    data = pd.DataFrame(data)
    data.sort_values(by='frame', inplace=True)
    data.to_hdf(args.outfile, 'segmented')


@timed
def batch_track(args):
    """outfile should have a `/segmented` table containing the segmentation data.

    saves the result in same file under `/tracked`
    """
    segments = pd.read_hdf(args.outfile, 'segmented')
    results = []
    tracker = SORTracker(min_dist=args.min_dist,
                         max_age=args.max_age,
                         n_init=args.min_hits,
                         min_hits=args.min_hits)
    for frame, fgrp in segments.groupby('frame'):
        if len(fgrp) == 0:
            continue
        tracked = tracker.update(fgrp.values)
        for tid, bbox in tracked.items():
            results.append({'frame': frame,
                            'trackid': tid,
                            'x': bbox[0],
                            'y': bbox[1],
                            'w': bbox[2],
                            'h': bbox[3]})

        if frame % 100 == 0:
            logging.info(f'Processed till {frame}')
    results = pd.DataFrame(results)
    results.sort_values(by='frame', inplace=True)
    results.to_hdf(args.outfile, 'tracked')


def make_parser():
    parser = argparse.ArgumentParser('Track objects in video in batch mode')
    parser.add_argument('-i', '--infile', type=str, help='input file')
    parser.add_argument('-o', '--outfile', type=str,
                        help='output file. Create an HDF file with segmentation'
                             ' in `segmented` and tracking data in the table'
                             ' `tracked`')
    parser.add_argument('-c', '--config', type=str,
                        help='configuration file to use for rest of the'
                             ' arguments')
    parser.add_argument('-m', '--seg_method', type=str, default='yolact',
                        help='method for segmentation'
                             ' (yolact/threshold/watershed/dbscan)')
    yolact_grp = parser.add_argument_group('YOLACT',
                                           'Parameters for YOLACT segmentation')
    yolact_grp.add_argument('--yconfig', type=str,
                            help='YOLACT configuration file')
    yolact_grp.add_argument('-w', '--weight', type=str,
                            help='YOLACT trained weights file')
    yolact_grp.add_argument('-s', '--score', type=float, default=0.3,
                            help='score threshold for accepting a detected object')
    yolact_grp.add_argument('-k', '--top_k', type=int, default=30,
                            help='maximum number of objects above score'
                                 ' threshold to keep')
    yolact_grp.add_argument('--cuda', type=bool, default=True,
                            help='Whether to use CUDA')
    thresh_grp = parser.add_argument_group(
        'Threshold',
        'Parameters for thresholding'
    )
    thresh_grp.add_argument('--thresh_method', type=str, default='gaussian',
                            help='Method for adaptive thresholding'
                                 ' (gaussian/mean)')
    thresh_grp.add_argument('--thresh_max', type=int, default=255,
                            help='Maximum intensity for thresholding')
    thresh_grp.add_argument('--thresh_baseline', type=int, default=10,
                            help='baseline intensity for thresholding')
    thresh_grp.add_argument('--blur_width', type=int, default=7,
                            help='blur width before thresholding.'
                                 ' Must be odd number.')
    thresh_grp.add_argument('--blur_sd', type=float, default=1.0,
                            help='SD for Gaussian blur before thresholding')
    thresh_grp.add_argument('--thresh_blocksize', type=int, default=41,
                            help='block size for adaptive thresholding.'
                                 ' Must be odd number')
    thresh_grp.add_argument('--invert_thresh', type=bool, default=True,
                            help='Inverted thresholding')
    watershed_grp = parser.add_argument_group(
        'Watershed',
        'Parameter for segmentation using watershed algorithm')
    watershed_grp.add_argument('-d', '--dist_thresh', type=float, default=3.0,
                               help='minimum distance of pixels from detected '
                                    'boundary to consider them core points')
    dbscan_grp = parser.add_argument_group(
        'DBSCAN',
        'Parameters for segmentation by clustering pixels with'
        ' DBSCAN algorithm')
    dbscan_grp.add_argument('-e', '--eps', type=float, default=5.0,
                            help='epsilon parameter for DBSCAN')
    dbscan_grp.add_argument('--min_samples', type=int, default=10,
                            help='minimum number of pixels in each cluster'
                                 ' for DBSCAN')
    limits_grp = parser.add_argument_group(
        'Limits',
        'Parameters to set limits on detected object size.')
    limits_grp.add_argument('--pmin', type=int, default=10,
                            help='Minimum number of pixels')
    limits_grp.add_argument('--pmax', type=int, default=500,
                            help='Maximum number of pixels')
    limits_grp.add_argument('--hmin', type=int, default=10,
                            help='Minimum height (longer side) of bounding box'
                                 ' in pixels')
    limits_grp.add_argument('--hmax', type=int, default=200,
                            help='Maximum height (longer side) of bounding box'
                                 ' in pixels')
    limits_grp.add_argument('--wmin', type=int, default=10,
                            help='Minimum width (shorter side) of bounding box'
                                 ' in pixels')
    limits_grp.add_argument('--wmax', type=int, default=100,
                            help='Maximum width (shorter side) of bounding box'
                                 ' in pixels')
    track_grp = parser.add_argument_group(
        'Tracker',
        'Parameters for SORT tracker')
    track_grp.add_argument('-x', '--overlap', type=float, default=0.3,
                           help='Minimum overlap between bounding boxes as a'
                                ' fraction of their total area.')
    track_grp.add_argument('--min_hits', type=int, default=3,
                           help='Minimum number of hits to accept a track')
    track_grp.add_argument('--max_age', type=int, default=50,
                           help='Maximum number of misses to exclude a track')
    parser.add_argument('--debug', action='store_true', help='Print debug info')
    return parser


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # 2 proc 40 / 124 fps
    # 5 proc 25 / 50 fps
    parser = make_parser()
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as cfg_file:
            config = yaml.safe_load(cfg_file)
            for key, value in config.items():
                vars(args)[key] = value
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    print('ARGS:')
    print(args)
    batch_segment(args)
    batch_track(args)
    print('Finished tracking')
