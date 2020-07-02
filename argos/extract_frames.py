#!/usr/bin/env python
# Author: SSubhasis Ray, ray dot subhasis at gmail dot com
# Date: 2020-02-23
"""Extract some frames from video as images"""
import sys
import os
import cv2
import argparse
import random


COLORMAPS = {'gray': cv2.COLOR_BGR2GRAY, 'hsv': cv2.COLOR_BGR2HSV, 'lab': cv2.COLOR_BGR2LAB}


def make_parser():
    parser = argparse.ArgumentParser('Extract frames from a video. '
                                     'If `-r` and `-n N` parameters are'
                                     ' specified, then dump `N` randomly'
                                     ' selected frames. If `-s START -i STRIDE`'
                                     ' are specified then dump every `STRIDE`-th'
                                     ' frame starting from `START` frame.')
    parser.add_argument('-f', dest='fname', type=str, help='input filename')
    parser.add_argument('-s', dest='start', default=0, type=int, help='starting frame')
    parser.add_argument('-i', dest='stride', default=1, type=int, help='stride, interval between successive frames to save.')
    parser.add_argument('-c', dest='cmap', default='', type=str, help='colormap to conevrt to, default same as original')
    parser.add_argument('-x', dest='scale', default=1, type=float, help='factor by which to scale the images')
    parser.add_argument('-r', dest='random', action='store_true', help='extract random frames')
    parser.add_argument('-n', dest='num', default=1, type=int, help='number of frames to extract. for use with `-r` option.')
    parser.add_argument('-o', dest='outdir', default='.', type=str, help='output directory')
    return parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.fname)
    fname = os.path.basename(args.fname)
    prefix = fname.rpartition('.')[0]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cmap = COLORMAPS.get(args.cmap, None)
    if args.random:
        if args.num < frame_count:
            frames = random.sample(range(frame_count), args.num)
        else:
            frames = range(frame_count)
    else:
        frames = range(args.start, frame_count, args.stride)
    for frame_no in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if frame is None:
            print('Could not read frame no:', frame_no)
            break
        image = cv2.cvtColor(frame, cmap) if cmap is not None else frame
        size = (int(frame.shape[1] * args.scale),
                int(frame.shape[0] * args.scale))
        if args.scale < 1:
            image = cv2.resize(image, size, cv2.INTER_AREA)
        elif args.scale > 1:
            image = cv2.resize(image, size, cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(args.outdir,
                                 f'{prefix}_{int(frame_no):06d}.png'), image)
    cap.release()
