# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-07-14 1:39 PM
"""Batch processing utility for tracking object in video.

This works using multiple processes to utilize multiple CPU cores.
"""
import os
from typing import Tuple
import sys
import logging
import threading
from functools import partial
import numpy as np
import cv2
import yaml
import time
import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from yolact import Yolact
from yolact.data import config as yconfig
# This is actually yolact.utils
from yolact.utils.augmentations import FastBaseTransform
from yolact.layers import output_utils as oututils

logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s '
                               'p=%(processName)s[%(process)d] '
                               't=%(threadName)s[%(thread)d] '
                               '%(filename)s#%(lineno)d:%(funcName)s: '
                               '%(message)s',
                        level=logging.DEBUG)

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def init_yolact(cfgfile, netfile):
    load_config(cfgfile)
    load_weights(netfile)


def process(frame_data: Tuple[int, np.ndarray], cuda: bool, score_threshold: float, top_k: int):
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

    pos, image = frame_data
    logging.debug(f'Received frame {pos}')
    if ynet is None:
        raise ValueError('Network not initialized')
    # Partly follows yolact eval.py
    tic = time.perf_counter_ns()
    with torch.no_grad():
        if cuda:
            image = torch.from_numpy(image).cuda().float()
        else:
            image = torch.from_numpy(image).float()
        batch = FastBaseTransform()(image.unsqueeze(0))
        preds = ynet(batch)
        image_gpu = image / 255.0
        h, w, _ = image.shape
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
        logging.debug('Time to process single _image: %f s',
                      1e-9 * (toc - tic))
        return (pos, boxes)
        logging.debug(f'Emitted bboxes for frame {pos}: {boxes}')


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


def load_weights(filename):
    global ynet
    if filename == '':
        raise ValueError('Empty filename for network weights')
    tic = time.perf_counter_ns()
    with torch.no_grad():
        if torch.cuda.is_available():
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

class BatchTrack(object):
    def __init__(self, video_filename, output_filename, config_filename,
                 weights_filename, score_threshold, top_k):
        self.batch_size = mp.cpu_count() * 4
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.video = cv2.VideoCapture(video_filename)
        if (self.video is None) or not self.video.isOpened():
            raise IOError('Could not open video')
        self.output_filename = output_filename
        self.cuda = False   # torch.cuda.is_available()
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.config_filename = config_filename
        self.weights_filename = weights_filename

    def read_frame(self):
        if (self.video is None) or not self.video.isOpened():
            return (None, -1)
        ret, frame = self.video.read()
        frame_no = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        logging.debug('Read frame no %d', frame_no)
        return (frame_no, frame)

    def start(self):
        vid_end = False
        process_partial = partial(process, cuda=self.cuda,
                                  score_threshold=self.score_threshold,
                                  top_k=self.top_k)
        results = {}
        with mp.Pool(initializer=init_yolact, initargs=(self.config_filename, self.weights_filename)) as pool:
            while True:
                frames = []
                for ii in range(self.batch_size):
                    frame_no, frame = self.read_frame()
                    if frame is None:
                        vid_end = True
                    frames.append((frame_no, frame))
                if len(frames) == 0:
                    break
                result = pool.imap_unordered(process_partial, frames,
                                            chunksize=10)
                for ret in result:
                    results[ret[0]] = ret[1]
                if vid_end:
                    break


if __name__ == '__main__':

    track = BatchTrack(video_filename='C:/Users/raysu/Documents/src/argos_data/dump/2020_02_20_00270.avi',
    output_filename='C:/Users/raysu/Documents/src/argos_data/dump/2020_02_20_00270.avi.track.csv',
                       config_filename='C:/Users/raysu/Documents/src/argos_data/yolact_annotations/yolact_config.yaml',
                       weights_filename='C:/Users/raysu/Documents/src/argos_data/yolact_annotations/babylocust_resnet101_119999_240000.pth',
                       score_threshold=0.15, top_k=10)
    track.start()




