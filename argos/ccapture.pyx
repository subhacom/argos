cimport cython
import sys
import os
import argparse
import time
import re
from datetime import datetime, timedelta
import csv
import numpy as np
import cv2
# import logging
# from logging.handlers import MemoryHandler
from matplotlib import colors


cimport numpy as cnp

import caputil


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint check_motion_rgb(cnp.ndarray[unsigned char, ndim=3] current,
                        cnp.ndarray[unsigned char, ndim=2] blurred_prev,
                        cnp.ndarray[unsigned char, ndim=2] blurred_cur,
                        int threshold,
                        int min_area,
                        int kernel_width=21,
                        bint show_diff=False):
    cdef cnp.ndarray[unsigned char, ndim=2] gray_cur = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    return check_motion_gray(gray_cur, blurred_prev, blurred_cur, threshold, min_area, kernel_width, show_diff)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint check_motion_gray(cnp.ndarray[unsigned char, ndim=2] gray_cur,
                        cnp.ndarray[unsigned char, ndim=2] blurred_prev,
                        cnp.ndarray[unsigned char, ndim=2] blurred_cur,
                        int threshold,
                        int min_area,
                        int kernel_width=21,
                        bint show_diff=False):
    blurred_cur[:, :] = cv2.GaussianBlur(gray_cur, (kernel_width,
                                              kernel_width), 0)

    cdef cnp.ndarray[unsigned char, ndim=2] frame_delta = cv2.absdiff(blurred_cur, blurred_prev)
    cdef cnp.int64_t ret
    contour_info = None
    contours = None
    hierarchy = None
    moving_contours = None
    ret, thresh_img = cv2.threshold(frame_delta, threshold, 255,
                                    cv2.THRESH_BINARY)
    thresh_img = cv2.dilate(thresh_img, None, iterations=2)
    # with opencv >= 3 findContours leaves the original image is
    # unmodified
    contour_info = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if caputil.CV2_MAJOR == 3:
        _, contours, hierarchy = contour_info
    elif caputil.CV2_MAJOR == 4:
        contours, hierarchy = contour_info
    moving_contours = [contour for contour in contours
                       if cv2.contourArea(contour) > min_area]
    if show_diff:
        win_diff_name = 'DEBUG - absdiff between frames'
        cv2.namedWindow(win_diff_name, cv2.WINDOW_NORMAL)
        win_thresh_name = 'DEBUG - thresholded absdiff'
        cv2.namedWindow(win_thresh_name, cv2.WINDOW_NORMAL)
        frame = cv2.applyColorMap(frame_delta, cv2.COLORMAP_JET)
        cv2.drawContours(frame, contours, -1, -1)
        cv2.imshow(win_diff_name, frame)
        cv2.imshow(win_thresh_name, thresh_img)
        
    if len(moving_contours) > 0:
        blurred_prev[:, :] = blurred_cur               
    return len(moving_contours) > 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int vcapture(str input_, str output, str fmt, int fps,
                    bint interactive, int width, int height,
                    int roi_x, int roi_y, int roi_w, int roi_h,
                    double interval,
                    double duration,
                    long max_frames,
                    bint motion_based,
                    bint config,
                    int threshold,
                    int min_area,
                    int kernel_width,                    
                    bint timestamp,
                    bint show_diff,
                    cnp.ndarray[int] tb,
                    cnp.ndarray[int] tc,
                    int tx,
                    int ty,
                    double fs
                    ):

    cdef int w_
    cdef int h_
    # cdef cnp.ndarray[unsigned char, ndim=2] frame 
    cdef cnp.ndarray[unsigned char, ndim=2] cur_blurred
    cdef cnp.ndarray[unsigned char, ndim=2] prev_blurred
    cdef int iinput = -1   # negative means file input
    cdef int file_count = 0
    cdef int read_frames = 0
    cdef int writ_frames = 0
    cdef bint save = True
    cdef bint vid_end = False
    cdef double ttot = 0.0
    cap = None
    tprev = None
    outfile = None
    tsfname = None
    tsfile = None
    if fps < 0:
        fps = 30
    try:
        iinput = int(input_)
        if (sys.platform == 'win32') and interactive:
            cap = cv2.VideoCapture(iinput, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(iinput)
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            cap.set(cv2.CAP_PROP_FPS, fps)
        if config:
            cap.set(cv2.CAP_PROP_SETTINGS, 1)
        print(f'Camera FPS set to {fps}')
        tstart = datetime.now()
        w_ = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h_ = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f'Frame size: {w_}x{h_}')
        if roi_x + roi_w > w_ or roi_y + roi_h > h_:
            print('ROI limits exceed frame size')
            return 0
    except ValueError:
        tstart = datetime.fromtimestamp(time.mktime(time.localtime(os.path.getmtime(input_))))
        cap = cv2.VideoCapture(input_)
        if not cap.isOpened() and (sys.platform == 'win32') and interactive:
            cap = cv2.VideoCapture(input_, cv2.CAP_DSHOW)
        print('Input from file', input_, 'Open?', cap.isOpened())
        if not cap.isOpened():
            print('Could not open video file'
                    f' {input_} for reading.')
            return 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('Frames to process', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
    try:
        while True:
            ret, frame = cap.read()
            if frame is None:
                break              
            if iinput >= 0:
                ts = datetime.now()
            else:
                ts = tstart + timedelta(days=0, seconds=read_frames / fps)
            w_ = frame.shape[1]
            h_ = frame.shape[0]
            if roi_x + roi_w > w_ or roi_y + roi_h > h_:
                print('ROI limits exceed frame size')
                return 0
            read_frames += 1
            
            if writ_frames == 0:
                fname, _, ext = output.rpartition('.')
                if max_frames > 0:
                    outfname = f'{fname}_{file_count:03d}.{ext}'
                else:
                    outfname = output
                while os.path.exists(outfname):
                    file_count += 1
                    outfname = f'{fname}_{file_count:03d}.{ext}'
                file_count += 1
                tsfname = f'{outfname}.csv'
                print(f'Video saved in {outfname}\n'
                      f'Timestamps in {tsfname}')
                tprev = ts                
                prev_blurred = np.zeros((roi_h, roi_w), dtype=np.uint8)
                cur_blurred = np.zeros((roi_h, roi_w), dtype=np.uint8) 
                fourcc = cv2.VideoWriter_fourcc(*fmt)
                if outfile is not None:
                    outfile.release()
                outfile = cv2.VideoWriter(outfname, fourcc, fps, (roi_w, roi_h))
                tsfile = open(tsfname, 'w', newline='')
                tswriter = csv.writer(tsfile)
                tswriter.writerow(['inframe', 'outframe', 'timestamp'])
                writ_frames = 1
                cv2.destroyAllWindows()
                if interactive:
                    cv2.namedWindow(outfname, cv2.WINDOW_NORMAL)
    
                # continue 

            ttot_ = ts - tstart
            ttot = ttot_.total_seconds()
            tdelta_ = ts - tprev            
            tdelta = tdelta_.total_seconds()
    
            if (duration > 0.0) and (ttot > duration):
                break
            roi = frame[roi_y: roi_y + roi_h,
                        roi_x: roi_x + roi_w].copy()
            if motion_based:
                if len(frame.shape) == 3:
                    save = check_motion_rgb(roi,
                                        prev_blurred,
                                        cur_blurred,
                                        threshold,
                                        min_area,
                                        kernel_width,
                                        show_diff)
                else:
                    print('Frame dimension', len(frame.shape))
                    save = check_motion_gray(roi,
                                        prev_blurred,
                                        cur_blurred,
                                        threshold,
                                        min_area,
                                        kernel_width,
                                        show_diff)
            elif interval > 0:
                save = tdelta >= interval
            
            if save:
                tprev = ts
                tstring = ts.isoformat()
                if timestamp:
                    (w, h), baseline = cv2.getTextSize(
                            tstring, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (tx, ty),
                                         (tx + w, ty - h),
                                         (tb[0], tb[1], tb[2]), cv2.FILLED)
                    cv2.putText(frame, tstring,
                                (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, fs,
                                (tc[0], tc[1], tc[2]), 2, cv2.LINE_AA)
            
                outfile.write(roi)
                tswriter.writerow([read_frames - 1, writ_frames - 1, tstring])
                if max_frames > 0:
                    writ_frames = (writ_frames + 1) % max_frames
                else:
                    writ_frames += 1
                if interactive:
                    cv2.imshow(outfname, roi)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                       break
    except KeyboardInterrupt:
        print('Caught keyboard interrupt')
    except FileNotFoundError:
        print('Could not find one or more file paths')
    finally:
        cap.release()
        if (outfile is not None) and outfile.isOpened():
            outfile.release()
        if tsfile is not None:
            tsfile.close()
        cv2.destroyAllWindows()
    return read_frames
