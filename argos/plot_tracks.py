# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-04 8:56 PM
"""Utility to display the tracks"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_tracks(filename):
    tracks = pd.read_csv(filename)
    # tracks.describe()
    for trackid, trackgrp in tracks.groupby('trackid'):
        pos = trackgrp.sort_values(by='frame')
        x = np.int0(pos.x + pos.w / 2.0)
        y = - np.int0(pos.y + pos.h / 2.0)
        plt.plot(x, y, '.-', label=str(trackid))
    plt.show()


def play_tracks(vidfile, trackfile):
    cap = cv2.VideoCapture(vidfile)
    tracks = pd.read_csv(trackfile)

    win = 'Tracked'
    for frame, trackdata in tracks.groupby('frame'):
        ret, frame = cap.read()
        if frame is None:
            break
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
    vidfile = 'C:/Users/raysu/Documents/src/argos_data/20200220_00270_motion_tracked.avi'
    trackfile = 'C:/Users/raysu/Documents/src/argos_data/20200220_00270_motion_tracked.avi.trk.csv'
    play_tracks(vidfile,
               trackfile)
    plot_tracks(trackfile)


