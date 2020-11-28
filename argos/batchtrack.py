# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-07-14 1:39 PM
"""Batch processing utility for tracking object in video.

This works using multiple processes to utilize multiple CPU cores.
"""
import os
import argparse
from typing import Tuple
import sys
import logging
import threading
from functools import partial
import numpy as np
import cv2
import yaml
import time
import pandas as pd
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
from argos.utility import match_bboxes

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


def init_yolact(cfgfile, netfile, cuda):
    load_config(cfgfile)
    load_weights(netfile, cuda)


def segment_yolact(frame: np.ndarray, cuda: bool,
            score_threshold: float, top_k: int):
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
        raise ValueError('Network not initialized')
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

    
def segment_classic():
    pass

    
def load_config(filename):
    global config
    config = yconfig.cfg
    if filename == '':
        return
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
            if np.any(np.isnan(prior)) or np.any(prior[:KalmanTracker.NDIM] < 0):
                logging.info(f'Found nan or negative in prior of {id_}')
                continue
            predicted_bboxes[id_] = prior[:KalmanTracker.NDIM]
        self.trackers = {id_: self.trackers[id_] for id_ in predicted_bboxes}
        for id_, bbox in predicted_bboxes.items():
            if np.any(bbox < 0):
                logging.debug(f'EEEE prediced bbox negative: {id_}: {bbox}')
        matched, new_unmatched, old_unmatched = match_bboxes(
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


class BatchTrack(object):
    def __init__(self, video_filename, output_filename,
                 hmin, hmax, wmin, wmax, min_iou, min_hits, max_age):
        """
        Parameters
        -------------
        video_filename: str
            path to video file to process
        output_filename: str
            path to output file. If the extension is ``csv`` then write in
            comma separated text file. If ``hdf`` or ``h5`` then write in hdf5
            format.
        hmin: int
            minimum height (longer side) of bounding box
        hmax: int
            maximum height (longer side) of bounding box
        wmin: int
            minimum width (shorter side) of bounding box
        wmax: int
            maximum width (shorter side) of bounding box
        min_iou: float
            minimum intersection over union of bboxes to be considered same object
        min_hits: int
            minimum number of detections before including object in track
        max_age: int
            maximum number of misses before discarding object        
        """
        self.video = cv2.VideoCapture(video_filename)
        if (self.video is None) or not self.video.isOpened():
            raise IOError(f'Could not open video "{video_filename}"')
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_filename = output_filename
        # tracker
        self.wmin = wmin
        self.wmax = wmax
        self.hmin = hmin
        self.hmax = hmax
        self.tracker = SORTracker()
        self.tracker.min_dist = 1 - min_iou
        self.tracker.max_age = max_age
        self.tracker.n_init = min_hits
        
    def read_frame(self):
        if (self.video is None) or not self.video.isOpened():
            return (None, -1)
        ret, frame = self.video.read()
        frame_no = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        logging.debug('Read frame no %d', frame_no)
        return (frame_no, frame)

        
        
class BatchTrackYolact(BatchTrack):
    def __init__(self, video_filename, output_filename, 
                 hmin, hmax, wmin, wmax, min_iou, min_hits, max_age,
                 config_filename, weights_filename, score_threshold, top_k,
                 cuda=None):
        """
        Parameters
        -------------
        video_filename: str
            path to video file to process
        output_filename: str
            path to output file. If the extension is ``csv`` then write in
            comma separated text file. If ``hdf`` or ``h5`` then write in hdf5
            format.
        min_iou: float
            intersection over union of bboxes to consider same objects.
        hmin: int
            minimum height (longer side) of bounding box
        hmax: int
            maximum height (longer side) of bounding box
        wmin: int
            minimum width (shorter side) of bounding box
        wmax: int
            maximum width (shorter side) of bounding box
        min_iou: float
            minimum intersection over union of bboxes to be considered same object
        min_hits: int
            minimum number of detections before including object in track
        max_age: int
            maximum number of misses before discarding object        
        config_filename: str
            Yolact configuration file path
        weights_filename: str
            path to file containing trained network weights
        score_threshold: float
            threshold for detection score
        top_k: int
            keep only the top ``top_k`` detections by score
        cuda: bool
            whether or not to use GPU. If ``None`` detect automatically.
        """
        super(BatchTrackYolact, self).__init__(
            video_filename, output_filename,
            hmin, hmax, wmin, wmax, min_iou, min_hits, max_age)
        if cuda is None:
            self.cuda = torch.cuda.is_available()
        else:
            self.cuda = cuda
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.config_filename = config_filename
        self.weights_filename = weights_filename
        init_yolact(self.config_filename, self.weights_filename, self.cuda)

    def process(self):
        results = []
        t0 = time.perf_counter_ns()
        while True:
            frame_no, frame = self.read_frame()
            t1 = time.perf_counter_ns()
            if frame is None:
                break
            bboxes = segment_yolact(frame,
                             cuda=self.cuda,
                             score_threshold=self.score_threshold,
                             top_k=self.top_k)
            t2 = time.perf_counter_ns()
            if len(bboxes) == 0:
                continue
            bboxes = bboxes[(bboxes[:, 2] >= self.wmin) &
                            (bboxes[:, 2] < self.wmax) &
                            (bboxes[:, 3] >= self.hmin) &
                            (bboxes[:, 3] < self.hmax)].copy()
                                    
            tracked = self.tracker.update(bboxes)
            for tid, bbox in tracked.items():
                results.append({'frame': frame_no,
                                'trackid': tid,
                                'x' : bbox[0],
                                'y': bbox[1],
                                'w': bbox[2],
                                'h': bbox[3]})
            t3 = time.perf_counter_ns()
            if frame_no % 100 == 0:
                logging.info(f'Processed till {frame_no} in {(t3 - t0) * 1e-9} seconds')
        results = pd.DataFrame(results)
        results = results.sort_values(by='frame')
        if self.output_filename.endswith('.csv'):
            results.to_csv(self.output_filename, index=False)
        elif self.output_filename.endswith(
                '.h5') or self.output_filename.endswith('.hdf'):
            results.to_hdf(self.output_filename, 'tracked')
        t4 = time.perf_counter_ns()
        logging.info(f'Saved {len(results)} bboxes '
                     f'in {(t4 - t3) * 1e-9} seconds.')


class BatchTrackClassic(BatchTrack):
    def __init__(self, video_filename, output_filename,
                 hmin, hmax, wmin, wmax, min_iou, min_hits, max_age,    
                 blur_width, blur_sd, 
                 invert_thresh, thresh_method, thresh_max, thresh_baseline,
                 thresh_blocksize, min_pixels, max_pixels):
        """
        Parameters
        -------------
        video_filename: str
            path to video file to process
        output_filename: str
            path to output file. If the extension is ``csv`` then write in
            comma separated text file. If ``hdf`` or ``h5`` then write in hdf5
            format.
        min_iou: float
            intersection over union of bboxes to consider same objects.
        hmin: int
            minimum height (longer side) of bounding box
        hmax: int
            maximum height (longer side) of bounding box
        wmin: int
            minimum width (shorter side) of bounding box
        wmax: int
            maximum width (shorter side) of bounding box
        min_iou: float
            minimum intersection over union of bboxes to be considered same object
        min_hits: int
            minimum number of detections before including object in track
        max_age: int
            maximum number of misses before discarding object        
        """
        pass
    

def make_parser():
    parser = argparse.ArgumentParser('Track objects in video in batch mode')
    parser.add_argument('-i', '--input', type=str, help='input file')
    parser.add_argument('-o', '--output', type=str, help='output file. extension .csv will create a text file with comma separated values, .h5 or .hdf will create an HDF file with data in the table `tracked`')
    parser.add_argument('-c', '--config', type=str, help='YOLACT configuration file')
    parser.add_argument('-w', '--weight', type=str, help='YOLACT trained weights file')
    parser.add_argument('-s', '--score', type=float, default=0.3, help='score threshold for accepting a detected object')
    parser.add_argument('-k', '--top_k', type=int, default=30, help='maximum number of objects above score threshold to keep')
    parser.add_argument('--hmin', type=int, default=10, help='Minimum height (longer side) of bounding box in pixels')
    parser.add_argument('--hmax', type=int, default=100, help='Maximum height (longer side) of bounding box in pixels')
    parser.add_argument('--wmin', type=int, default=10, help='Minimum width (shorter side) of bounding box in pixels')
    parser.add_argument('--wmax', type=int, default=100, help='Maximum width (shorter side) of bounding box in pixels')
    parser.add_argument('-x', '--overlap', type=float, default=0.3, help='Minimum overlap between bounding boxes as a fraction of their total area.')
    parser.add_argument('--min_hits', type=int, default=3, help='Minimum number of hits to accept a track')
    parser.add_argument('--max_age', type=int, default=50, help='Maximum number of misses to exclude a track')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use CUDA')
    parser.add_argument('--debug', action='store_true', help='Print debug info')
    return parser

    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # 2 proc 40 / 124 fps
    # 5 proc 25 / 50 fps
    parser = make_parser()
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    print('ARGS:')
    print(args)
    tracker = BatchTrackYolact(
        video_filename=args.input,        
        output_filename=args.output,
        config_filename=args.config,
        weights_filename=args.weight,
        score_threshold=args.score,
        top_k=args.top_k,
        cuda=args.cuda,
        hmin=args.hmin,
        hmax=args.hmax,
        wmin=args.wmin,
        wmax=args.wmax,
        min_iou=args.overlap,
        min_hits=args.min_hits,
        max_age=args.max_age)
    tracker.process()
