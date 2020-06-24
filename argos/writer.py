# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-03 12:07 AM
import logging
import os
from collections import OrderedDict
import csv
import numpy as np
from PyQt5 import QtCore as qc
import pandas as pd
from argos.utility import init


settings = init()

segmented_key = 'segmented'
tracked_key = 'tracked'


def makepath(dirname, fname):
    fname = os.path.basename(fname)
    filename = os.path.join(dirname, f'{fname}.h5')
    ii = 0
    while os.path.exists(filename):
        ii += 1
        filename = os.path.join(dirname, f'{fname}.{ii}.h5')
    return filename


class DataHandler(qc.QObject):
    def __init__(self, filename, mode='w'):
        super(DataHandler, self).__init__()
        self.filename = filename
        self.mode = mode
        self.data_store = pd.HDFStore(filename, mode=mode,
                                      complib='blosc')
        # Buffer the data in OrderedDicts keyed by frame num.
        self._segmented = OrderedDict()
        self._tracked = OrderedDict()
        try:
            seg = self.data_store.get(segmented_key)
            for frame_no, grp in seg.groupby('frame'):
                self._segmented[frame_no] = grp[['x', 'y', 'w', 'h']].values
        except KeyError:
            logging.debug(f'No segmentation data in {self.filename}')
        try:
            trk = self.data_store.get(tracked_key)
            for frame_no, grp in trk.groupby('frame'):
                self._tracked[frame_no] = grp[['trackid',
                                              'x', 'y', 'w', 'h']].values
        except KeyError:
            logging.debug(f'No tracking data in {self.filename}')
        assert len(self._segmented) == len(self._tracked),  \
        'Segmented and tracked frame count are different. Corrupt data file?'

    @qc.pyqtSlot(np.ndarray, int)
    def appendSegmented(self, bboxes: np.ndarray, frame_no: int):
        self._segmented[frame_no] = bboxes

    @qc.pyqtSlot(dict, int)
    def appendTracked(self, id_bbox: dict, frame_no: int):
        self._tracked[frame_no] = id_bbox

    def _write(self):
        """
        Not going to append, for even long videos the numbers should be
        small enough to fit in the RAM of a half-decent laptop.
        Writing should happen only at the end.
        """
        data = []
        for frame_no, seg in self._segmented.items():
            if len(seg) == 0:
                continue
            data.append(np.c_[[frame_no] * seg.shape[0], seg])
        if len(data) == 0:
            logging.info('No segmentation data. Not writing file.')
            return
        data = np.concatenate(data)
        data = pd.DataFrame(data=data, columns=['frame', 'x', 'y', 'w', 'h'])
        self.data_store.put(segmented_key, data, format='table', append=False)
        data = []
        for frame_no, trk in self._tracked.items():
            if len(trk) == 0:
                continue
            ids = np.array(list(trk.keys()))
            pos = np.array(list(trk.values()))
            data.append(np.c_[[frame_no] * len(trk), ids, pos])
        if len(data) == 0:
            logging.info('No tracking data.')
            return
        data = np.concatenate(data)
        data = pd.DataFrame(data=data, columns=['frame', 'trackid',
                                                'x', 'y', 'w', 'h'])
        self.data_store.put(tracked_key, data, format='table', append=False)

    @qc.pyqtSlot()
    def close(self):
        if self.data_store.is_open:
            self._write()
            self.data_store.close()
        self._segmented = OrderedDict()
        self._tracked = OrderedDict()

    @qc.pyqtSlot()
    def reset(self):
        """Overwrite current files, going back to start."""
        self.close()
        self.data_store = pd.HDFStore(self.filename,
                                      mode=self.mode,
                                      complib='blosc')

    def __del__(self):
        self.close()


class CSVWriter(qc.QObject):
    """Not used any more. Using HDF5 is much faster"""
    def __init__(self):
        super(CSVWriter, self).__init__()
        self.out_dir = '.'
        self.seg_filename = ''
        self.track_filename = ''
        self.seg_file = None
        self.track_file = None
        self.seg_writer = None
        self.track_writer = None

    def set_path(self, directory, fileprefix):
        self.seg_filename = os.path.join(
            directory,
            f'{os.path.basename(fileprefix)}.seg.csv')
        ii = 0
        while os.path.exists(self.seg_filename):
            ii += 1
            self.seg_filename = os.path.join(
                directory,
                f'{os.path.basename(fileprefix)}.seg{ii}.csv')

        if self.seg_file is not None and not self.seg_file.closed:
            self.seg_file.close()
        self.seg_file = open(self.seg_filename, 'w', newline='')
        self.seg_writer = csv.writer(self.seg_file)
        self.seg_writer.writerow('frame,x,y,w,h'.split(','))
        self.track_filename = os.path.join(
                directory,
                '{}.track{}.csv'.format(os.path.basename(fileprefix),
                    '' if ii == 0 else ii))
        if os.path.exists(self.track_filename):
            raise Exception(f'File already exists {self.track_filename}')
        if self.track_file is not None and not self.track_file.closed:
            self.track_file.close()
        self.track_file = open(self.track_filename, 'w', newline='')
        self.track_writer = csv.writer(self.track_file)
        self.track_writer.writerow('frame,trackid,x,y,w,h'.split(','))

    @qc.pyqtSlot(np.ndarray, int)
    def writeSegmented(self, bboxes: np.ndarray, frame_no: int):
        for bbox in bboxes:
            data = [frame_no] + list(bbox)
            self.seg_writer.writerow(data)

    @qc.pyqtSlot(dict, int)
    def writeTracked(self, id_bbox: dict, frame_no: int):
        for id_ in sorted(id_bbox):
            data = [frame_no, id_] + list(id_bbox[id_])
            self.track_writer.writerow(data)

    @qc.pyqtSlot()
    def close(self):
        if self.seg_file is not None and not self.seg_file.closed:
            self.seg_file.close()
        if self.track_file is not None and not self.track_file.closed:
            self.track_file.close()

    @qc.pyqtSlot()
    def reset(self):
        """Overwrite current files, going back to start"""
        self.close()
        self.seg_file = open(self.seg_filename, 'w', newline='')
        self.seg_writer = csv.writer(self.seg_file)
        self.seg_writer.writerow('frame,x,y,w,h'.split(','))
        self.track_file = open(self.track_filename, 'w', newline='')
        self.track_writer = csv.writer(self.track_file)
        self.track_writer.writerow('frame,trackid,x,y,w,h'.split(','))

    def __del__(self):
        self.close()