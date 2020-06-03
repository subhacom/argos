# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2020-06-03 12:07 AM
import os
import csv
import numpy as np
from PyQt5 import QtCore as qc


class Writer(qc.QObject):
    def __init__(self):
        super(Writer, self).__init__()
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

    @qc.pyqtSlot(np.ndarray, int)
    def writeSegmented(self, bboxes, frame):
        for bbox in bboxes:
            data = [frame] + list(bbox)
            self.seg_writer.writerow(data)

    @qc.pyqtSlot(dict, int)
    def writeTracked(self, id_bbox, frame):
        for id_ in sorted(id_bbox):
            data = [frame, id_] + list(id_bbox[id_])
            self.track_writer.writerow(data)

    def close(self):
        self.seg_file.close()
        self.track_file.close()

    def __del__(self):
        if self.seg_file is not None and not self.seg_file.closed:
            self.seg_file.close()
        if self.track_file is not None and not self.track_file.closed:
            self.track_file.close()
