# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-04 8:56 PM
"""
=============================
Utility to display the tracks
=============================

Usage:
::
    python -m argos.plot_tracks -v {videofile} -f {trackfile} \\
    --torig {original-timestamps-file} \\
    --tmt {motiontracked-timestamps-file} \\
    --fplot {plotfile} \\
    --vout {video-output-file}

Try ``python -m argos.plot_tracks -h`` for a listing of all the
command line options.

This program allows displaying the (possibly motion-tracked) video
with the bounding boxes and IDs of the tracked objects overlaid.
Finally, it plots the tracks over time, possibly on a frame of the
video.

With ``--torig`` and ``--tmt`` options it will try to read the
timestamps from these files, which should have comma separated values
(.csv) with the columns ``inframe, outframe, timestamp`` (If you use
:py:module:``argos.capture`` to capture video, these will be aleady
generated for you). The frame-timestamp will be displayed on each
frame in the video. It will also be color-coded in the plot by
default.

With the ``--fplot`` option, it will save the plot in the filename
passed after it.

With the ``--vout`` option, it will save the video with bounding boxes
in the filename passed after it.

"""
import os
import cv2
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils.murmurhash import murmurhash3_32
from argparse import ArgumentParser


def plot_tracks(trackfile, ms=5, lw=5, show_bbox=True,
                bbox_alpha=(0.0, 1.0), plot_alpha=1.0, quiver=True,
                qcmap='hot', vidfile=None, frame=-1, gray=False,
                axes=False):
    if trackfile.endswith('.csv'):
        tracks = pd.read_csv(trackfile)
    else:
        tracks = pd.read_hdf(trackfile, 'tracked')
    tracks.describe()
    print('%%%%', bbox_alpha)
    img = None
    fig, ax = plt.subplots()
    if vidfile is not None:
        cap = cv2.VideoCapture(vidfile)
        if frame < 0:
            frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ret, img = cap.read()
        if img is None:
            print('Could not read image')
        elif img.shape[-1] == 3:  # BGR
            if gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            gray = True
    if img is not None:
        if gray:
            ax.imshow(img, origin='upper', cmap='gray')
        else:
            ax.imshow(img, origin='upper')
        if not axes:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            [ax.spines[s].set_visible(False)
             for s in ['left', 'bottom', 'top', 'right']]
        
    for trackid, trackgrp in tracks.groupby('trackid'):
        pos = trackgrp.sort_values(by='frame')
        cx = pos.x + pos.w / 2.0
        # The Y axis is inverted when using image.
        # Keep it consistent when no image is used.
        if img is None:
            cy = - (pos.y + pos.h / 2.0)
        else:
            cy = pos.y + pos.h / 2.0

        val = murmurhash3_32(int(trackid), positive=True).to_bytes(8, 'little')
        color = (val[0] / 255.0, val[1] / 255.0, val[2] / 255.0)
        if show_bbox:
            alpha = np.linspace(bbox_alpha[0], bbox_alpha[1], len(pos))
            ii = 0
            for p in pos.itertuples():
                bbox = plt.Rectangle((p.x, p.y),
                                     p.w, p.h,
                                     linewidth=lw,
                                     edgecolor=color,
                                     facecolor='none',
                                     alpha=alpha[ii])
                ii += 1
                ax.add_patch(bbox)

        if quiver:
            u = np.diff(cx)
            v = np.diff(cy)
            c = np.linspace(0, 1, len(u))
            ax.quiver(cx[:-1], cy[:-1], u, v, c, scale_units='xy', angles='xy',
                      scale=1, cmap=qcmap)
        else:
            plt.plot(cx, cy, '.-', color=color, ms=ms, alpha=plot_alpha,
                     label=str(trackid))
    fig.tight_layout()
    return fig


def play_tracks(vidfile, trackfile, lw=2, color='auto',
                fontscale=1, fthickness=1,
                torigfile=None, tmtfile=None,
                vout=None, outfmt='MJPG', fps=None,
                timestamp=False, skipempty=False):
    """
    skipempty: bool
        skip frames without any track 
    """
    cap = cv2.VideoCapture(vidfile)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        print('Could not open file', vidfile)
    if trackfile.endswith('.csv'):
        tracks = pd.read_csv(trackfile)
    else:
        tracks = pd.read_hdf(trackfile, 'tracked')
    timestamps = None
    if torigfile is not None:
        torig = pd.read_csv(torigfile)
        timestamps = torig
    if tmtfile is not None:
        tmt = pd.read_csv(tmtfile)
        timestamps = pd.merge(torig, tmt, left_on='outframe',
                              right_on='inframe')
        timestamps.drop(['outframe_x', 'timestamp_y', 'inframe_x',
                         'inframe_y'],
                        axis=1, inplace=True)
        timestamps.rename({'inframe_x': 'origframe', 'inframe_y': 'inframe',
                           'timestamp_x': 'timestamp', 'outframe_y': 'frame'},
                          axis=1,
                          inplace=True)

    if timestamps is None:
        tstart = datetime.fromtimestamp(time.mktime(time.localtime(
            os.path.getmtime(vidfile))))
        infps = cap.get(cv2.CAP_PROP_FPS)
        dt = np.arange(frame_count) / infps
        ts = tstart + pd.to_timedelta(dt, unit='s')
        timestamps = pd.DataFrame({'frame': np.arange(frame_count),
                                   'timestamp': ts})
    else:
        timestamps['timestamp'] = pd.to_datetime(timestamps['timestamp'])
        tstart = timestamps['timestamp'].min()
    win = os.path.basename(vidfile)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    colors = {}
    for ii in set(tracks.trackid.values):
        if color == 'auto':
            val = murmurhash3_32(int(ii), positive=True).to_bytes(8, 'little')
            colors[ii] = (val[0], val[1], val[2])

        else:
            colors[ii] = (0, 0, 255)
    out = None
    if vout is not None:
        fourcc = cv2.VideoWriter_fourcc(*outfmt)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps is None:
            fps = infps
        out = cv2.VideoWriter(vout, fourcc, fps,
                              (width, height))
        print(f'Saving video with tracks in {vout}. Video format {outfmt}')
    frame_no = -1
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('End at frame', frame_no)
            break
        frame_no += 1
        trackdata = tracks[tracks.frame == frame_no]
        if (len(trackdata) == 0) and skipempty:
            continue
        if timestamp:
            cv2.putText(frame, str(int(frame_no)), (100, 100),
                        cv2.FONT_HERSHEY_COMPLEX, fontscale, (255, 255, 0),
                        fthickness, cv2.LINE_AA)
            ts = timestamps[timestamps['frame'] == frame_no]['timestamp'].iloc[0]
            cv2.putText(frame, str(ts), (frame.shape[1] - 200, 100),
                        cv2.FONT_HERSHEY_COMPLEX, fontscale, (255, 255, 0),
                        fthickness, cv2.LINE_AA)
        for row in trackdata.itertuples():
            # print(f'{row.x}\n{row.y}\n{row.w}\n=====')
            id_ = int(row.trackid)
            print(id_, colors[id_])
            cv2.rectangle(frame, (int(row.x), int(row.y)),
                          (int(row.x + row.w), int(row.y + row.h)),
                          colors[id_], lw)
            cv2.putText(frame, str(id_), (int(row.x), int(row.y)),
                        cv2.FONT_HERSHEY_COMPLEX, fontscale, colors[id_],
                        fthickness, cv2.LINE_AA)
        cv2.imshow(win, frame)
        if out is not None:
            out.write(frame)
        key = cv2.waitKey(100)
        if key == ord('q') or key == 27:
            break
    if out is not None:
        out.release()
    cap.release()


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('-v', '--video', type=str, required=False, default='',
                        help='Video file from which tracking was done')
    parser.add_argument('-f', '--data', type=str, required=True,
                        help='Tracked data file')
    parser.add_argument('--torig', type=str,
                        help='Timestamp file for original video')
    parser.add_argument('--tmt', type=str,
                        help='Timestamp file for video with movement'
                        ' detection')
    parser.add_argument('-q', '--quiver', action='store_true',
                        help='Show quiver plot for tracks')
    parser.add_argument('--qcmap', default='hot',
                        help='Colormap used in quiver plot. This can be any'
                        ' colormap name defined in the matplotlib library')
    parser.add_argument('-b', '--bbox', action='store_true',
                        help='Show bounding boxes of tracked objects')
    parser.add_argument('--af', default=0.0, type=float,
                        help='When displaying bounding box, alpha value at'
                        ' first frame.')
    parser.add_argument('--al', default=1.0, type=float,
                        help='When displaying bounding box, alpha value at'
                        ' last frame.')
    parser.add_argument('--ap', default=1.0, type=float,
                        help='Alpha value of plot lines')
    parser.add_argument('--wp', default=8.0, type=float,
                        help='Width of plot figure in inches')
    parser.add_argument('--hp', default=6.0, type=float,
                        help='Height of plot figure in inches')
    parser.add_argument('--ms', default=1, type=int,
                        help='Marker size of plot lines')
    parser.add_argument('--vlw', default=2, type=int,
                        help='Line width of bounding box in video')
    parser.add_argument('--plw', default=2, type=int,
                        help='Line width of bounding box in plot')
    parser.add_argument('--fs', default=1.0, type=float,
                        help='Font scale to use in video display')
    parser.add_argument('--ft', default=10, type=int,
                        help='Font thickness to use in video display')
    parser.add_argument('--fps', default=10, type=int,
                        help='Frames per second in output video')
    parser.add_argument('--bgframe', default=0, type=int,
                        help='Display tracks on background with frame #')
    parser.add_argument('--fplot', type=str,
                        help='Save plot in this file')
    parser.add_argument('--vout', type=str,
                        help='Save video in this file')
    parser.add_argument('--vfmt', type=str, default='MJPG',
                        help='Output video format')
    parser.add_argument('--gray', action='store_true',
                        help='Make plot background image gray')    
    parser.add_argument('--timestamp', action='store_true',
                        help='Enable time stamp and frame no in video display')
    parser.add_argument('--axes', action='store_true',
                        help='Show axes in plot overlayed on image')
    parser.add_argument('--play', action='store_true',
                        help='Play video with bounding boxes')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    if (args.video is not None) and args.play:
        play_tracks(args.video, args.data, lw=args.vlw, fontscale=args.fs,
                    fthickness=args.ft,
                    torigfile=args.torig,
                    tmtfile=args.tmt,
                    vout=args.vout,
                    outfmt=args.vfmt, fps=args.fps, timestamp=args.timestamp)
    fig = plot_tracks(args.data, vidfile=args.video, ms=args.ms,
                      show_bbox=args.bbox, bbox_alpha=(args.af, args.al),
                      plot_alpha=args.ap,
                      quiver=args.quiver, qcmap=args.qcmap,
                      frame=args.bgframe,
                      gray=args.gray,
                      axes=args.axes)
    fig.set_size_inches(args.wp, args.hp)
    if args.fplot is not None:
        fig.savefig(args.fplot, transparent=True)
    plt.show()
