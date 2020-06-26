# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-04 8:56 PM
"""Utility to display the tracks"""
import sys
import cv2
import numpy as np
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


def play_tracks(vidfile, trackfile):
    cap = cv2.VideoCapture(vidfile)
    if not cap.isOpened():
        print('Could not open file', vidfile)
    if trackfile.endswith('.csv'):
        tracks = pd.read_csv(trackfile)
    else:
        tracks = pd.read_hdf(trackfile, 'tracked')

    win = 'Tracked'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for frame_no, trackdata in tracks.groupby('frame'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ret, frame = cap.read()
        if frame is None:
            print('End at frame', frame_no)
            break
        cv2.putText(frame, str(int(frame_no)), (100, 100),
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 0), 2)
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
    play_tracks(vidfile,
               trackfile)
    plot_tracks(trackfile)


