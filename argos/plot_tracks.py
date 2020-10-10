# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-04 8:56 PM
"""Utility to display the tracks"""
import os
import sys
import cv2
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
import pandas as pd


def plot_tracks(trackfile):
    if trackfile.endswith('.csv'):
        tracks = pd.read_csv(trackfile)
    else:
        tracks = pd.read_hdf(trackfile, 'tracked')
    # tracks.describe()
    for trackid, trackgrp in tracks.groupby('trackid'):
        pos = trackgrp.sort_values(by='frame')
        x = np.int0(pos.x + pos.w / 2.0)
        y = - np.int0(pos.y + pos.h / 2.0)
        plt.plot(x, y, '.-', alpha=0.5, label=str(trackid))
    plt.show()


def play_tracks(vidfile, trackfile, torigfile=None, tmtfile=None):
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
        timestamps = pd.merge(torig, tmt, left_on='outframe', right_on='inframe')
        timestamps.drop(['outframe_x', 'timestamp_y', 'inframe_x', 'inframe_y'], axis=1, inplace=True)
        timestamps.rename({'inframe_x': 'origframe', 'inframe_y': 'inframe',
                           'timestamp_x': 'timestamp', 'outframe_y': 'frame'},
                          axis=1,
                          inplace=True)

    if timestamps is None:
        tstart = datetime.fromtimestamp(time.mktime(time.localtime(
            os.path.getmtime(vidfile))))
        fps = cap.get(cv2.CAP_PROP_FPS)
        dt = np.arange(frame_count) / fps
        ts = tstart + pd.to_timedelta(dt, unit='s')
        timestamps = pd.DataFrame({'frame': np.arange(frame_count),
                                   'timestamp': ts})
    else:
        timestamps['timestamp'] = pd.to_datetime(timestamps['timestamp'])
        tstart = timestamps['timestamp'].min()
    win = os.path.basename(vidfile)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for frame_no, trackdata in tracks.groupby('frame'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ret, frame = cap.read()
        if frame is None:
            print('End at frame', frame_no)
            break
        cv2.putText(frame, str(int(frame_no)), (100, 100),
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 0), 2)
        ts = timestamps[timestamps['frame'] == frame_no]['timestamp'].iloc[0]
        cv2.putText(frame, str(ts), (frame.shape[1] - 200, 100),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 2)
        for idx, row in trackdata.iterrows():
            # print(f'{row.x}\n{row.y}\n{row.w}\n=====')
            cv2.rectangle(frame, (int(row.x), int(row.y)), (int(row.x + row.w), int(row.y + row.h)),
                          (0, 0, 255), 2)
            cv2.putText(frame, str(int(row.trackid)), (int(row.x), int(row.y)),
                                                cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 0), 2)
        cv2.imshow(win, frame)
        key = cv2.waitKey(100)
        if key == ord('q') or key == 27:
            break
    cap.release()


if __name__ == '__main__':
    vidfile = sys.argv[1]
    trackfile = sys.argv[2]
    torig = sys.argv[3] if len(sys.argv) > 3 else None
    tmt = sys.argv[4] if len(sys.argv) > 4 else None
    play_tracks(vidfile,
               trackfile)
    plot_tracks(trackfile)


