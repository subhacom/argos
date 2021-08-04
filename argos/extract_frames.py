#!/usr/bin/env python
# Author: SSubhasis Ray, ray dot subhasis at gmail dot com
# Date: 2020-02-23
"""
==========================================
Extract some frames from a video as images
==========================================

This is a little utility to extract frames from a video file.

try ``python -m argos.extract_frames -h`` to see possible command-line
arguments.

"""

import sys
import os
import cv2
import argparse
import random


COLORMAPS = {'gray': cv2.COLOR_BGR2GRAY,
             'hsv': cv2.COLOR_BGR2HSV,
             'lab': cv2.COLOR_BGR2LAB}


def make_parser():
    parser = argparse.ArgumentParser(
        'Extract frames from a video. '
        'If `-r` and `-n N` parameters are'
        ' specified, then dump `N` randomly'
        ' selected frames. If `-s START -i STRIDE`'
        ' are specified then dump every `STRIDE`-th'
        ' frame starting from `START` frame.')
    parser.add_argument('-f', dest='fname', type=str, help='input filename')
    parser.add_argument('-s', dest='start', default=0, type=int,
                        help='starting frame')
    parser.add_argument('-i', dest='stride', default=1, type=int,
                        help='stride, interval between successive frames'
                        ' to save.')
    parser.add_argument('-c', dest='cmap', default='', type=str,
                        help='colormap to conevrt to, default '
                        'same as original')
    parser.add_argument('-x', dest='scale', default=1, type=float,
                        help='factor by which to scale the images')
    parser.add_argument('-r', dest='random', action='store_true',
                        help='extract random frames')
    parser.add_argument('-n', dest='num', default=-1, type=int,
                        help='number of frames to extract.')
    parser.add_argument('-o', dest='outdir', default='.', type=str,
                        help='output directory')
    return parser


def convert_and_write(frame, scale, cmap, outfile):
    image = cv2.cvtColor(frame, cmap) if cmap is not None else frame
    size = (int(frame.shape[1] * scale),
            int(frame.shape[0] * scale))
    if scale < 1:
        image = cv2.resize(image, size, cv2.INTER_AREA)
    elif scale > 1:
        image = cv2.resize(image, size, cv2.INTER_CUBIC)
    cv2.imwrite(outfile, image)
    

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.fname)
    fname = os.path.basename(args.fname)
    prefix = fname.rpartition('.')[0]
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cmap = COLORMAPS.get(args.cmap, None)
    if args.num < 0:
        end = frame_count
    else:
        end = args.start + args.num * args.stride
    if end > frame_count:
        print(f'The last frame no {end} is'
              f' greater than frame count {frame_count}.'
              f' (start + num * stride) must be less than frame count.')
        sys.exit(1)
        
    if args.random:
        frames = random.sample(range(args.start, end), args.num)
    else:
        frames = range(args.start, end, args.stride)
    frames = set(frames)
    print(f'Going to extract {len(frames)} frames from {fname}'
          f' with total {frame_count} frames', end=' ')
    if args.random and args.num < frame_count:
        print('randomly selected')
    else:
        print(f'sequentially, starting at '
              f'{args.start} with {args.stride} stride')
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != args.start:
        cap.release()
        cv2.VideoCapture(args.fname)
        print('Cannot seek randomly in video. Reading sequentially')
        for frame_no in range(frame_count):
            ret, frame = cap.read()
            if frame is None:
                print('Could not read frame no:', frame_no)
                break
            if frame_no in frames:
                outfile = os.path.join(args.outdir,
                                       f'{prefix}_{int(frame_no):06d}.png')
                convert_and_write(frame, args.scale, cmap, outfile)
    else:
        for frame_no in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if frame is None:
                print('Could not read frame no:', frame_no)
                break
            outfile = os.path.join(args.outdir,
                                 f'{prefix}_{int(frame_no):06d}.png')
            convert_and_write(frame, args.scale, cmap, outfile)
    cap.release()
