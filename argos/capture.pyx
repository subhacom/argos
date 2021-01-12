import sys
import os
import argparse
import time
import re
from datetime import datetime, timedelta
import csv
import cv2
# import logging
# from logging.handlers import MemoryHandler
from matplotlib import colors


cdef void capture(str input_, str output, str format,
     bint interactive, int width, int height,
     int roi_x, int roi_y, int roi_w, int roi_h,
     long interval):

    cdef int prev_frame = -1
    cdef char * outname = output
    cdef int iinput = -1   # negative means file
    cdef int file_count = 0
    cdef int read_frames = 0
    cdef int writ_frames = 0
    cdef int fps
    cdef bint save = True
    cdef bint vid_end = False
    t_prev = None

    if interactive:
        cv2.namedWindow(outname, cv2.WINDOW_NORMAL)
    if (sys.platform == 'win32') and isinstance(input_, int):
        cap = cv2.VideoCapture(input_, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(input_)
    try:
        iinput = int(input_)
	if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	 if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	tstart = datetime.now()
	print('Input from camera', input_)
	print(f'Suggested size: {params["width"]}x{params["height"]}')
        print(f'Actual size: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x'
              f'{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
    except ValueError:
        tstart = datetime.fromtimestamp(time.mktime(time.localtime(os.path.getmtime(input_))))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('Input from file', input_)

    while not vid_end:
        ret, frame = cap.read()
        if frame is None:
            break
