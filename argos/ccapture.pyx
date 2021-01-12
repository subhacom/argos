cimport cython
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


cimport numpy as np


CV2_MAJOR, CV2_MINOR, _ = cv2.__version__.split(".")
CV2_MAJOR = int(CV2_MAJOR)
CV2_MINOR = int(CV2_MINOR)
cpdef int LARGE_FRAME_SIZE = 10000


def get_roi(input_, width, height):
    roi = 'Select ROI. Press ENTER when done, C to try again'
    cv2.namedWindow(roi, cv2.WINDOW_NORMAL)
    x, y, w, h = 0, 0, 0, 0
    try:
        input_ = int(input_)
    except ValueError:
        pass
    capture = cv2.VideoCapture(input_)
    if not capture.isOpened():
        print(f'Could not open video {input_}')
        return None
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while True:
        ret, frame = capture.read()
        x, y, w, h = cv2.selectROI(roi, frame)
        print('ROI', x, y, w, h)
        if w > 0 and h > 0:
            break
    capture.release()
    cv2.destroyAllWindows()
    return (int(x), int(y), int(w), int(h), int(width), int(height))
        

def get_camera_fps(devid, width, height, fps=30, nframes=120):
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(devid, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(devid)
    start = datetime.now()
    assert cap.isOpened(), 'Could not open camera'
    if width < 0 or height < 0:
        width = LARGE_FRAME_SIZE
        height = LARGE_FRAME_SIZE
        print('Trying maximum possible resolution')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for ii in range(nframes):
        ret, frame = cap.read()
    cap.release()
    end = datetime.now()
    delta = end - start
    interval = delta.seconds + delta.microseconds * 1e-6
    fps = nframes / interval
    return fps, width, height

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef bint check_motion(np.ndarray[unsigned char, ndim=3] current,
                        np.ndarray[unsigned char, ndim=3] prev,
                        int threshold,
                        int min_area,
                        int kernel_width=21):
    cdef np.ndarray[unsigned char, ndim=2] gray_cur = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.GaussianBlur(gray_cur, (kernel_width,
                                           kernel_width), 0)
    cdef np.ndarray[unsigned char, ndim=2] gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.GaussianBlur(gray_prev, (kernel_width,
                                             kernel_width), 0)
    cdef np.ndarray[unsigned char, ndim=2] frame_delta = cv2.absdiff(gray_cur, gray_prev)
    cdef np.int_t ret
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
    if CV2_MAJOR == 3:
        _, contours, hierarchy = contour_info
    elif CV2_MAJOR == 4:
        contours, hierarchy = contour_info
    moving_contours = [contour for contour in contours
                       if cv2.contourArea(contour) > min_area]
    return len(moving_contours) > 0


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef void vcapture(str input_, str output, str fmt, int fps,
                    bint interactive, int width, int height,
                    int roi_x, int roi_y, int roi_w, int roi_h,
                    long interval,
                    double duration,
                    long max_frames,
                    bint motion_based=False,
                    int threshold=100,
                    int min_area=100,
                    int kernel_width=21):

    cdef int w_
    cdef int h_
    cdef np.ndarray[unsigned char, ndim=3] prev_frame
    cdef int iinput = -1   # negative means file input
    cdef int file_count = 0
    cdef int read_frames = 0
    cdef int writ_frames = 0
    cdef bint save = True
    cdef bint vid_end = False
    cdef double ttot = 0.0
    tprev = None
    prev_frame = None
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
        print(f'Camera FPS set to {fps}')
        tstart = datetime.now()
        print('Input from camera', input_)
        print(f'Suggested size: {width}x{height}')
        w_ = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h_ = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f'Actual size: {w_}x{h_}')
        if roi_x + roi_w > w_ or roi_y + roi_h > h_:
            print('ROI limits exceed frame size')
            return
    except ValueError:
        tstart = datetime.fromtimestamp(time.mktime(time.localtime(os.path.getmtime(input_))))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if (sys.platform == 'win32') and interactive:
            cap = cv2.VideoCapture(input_, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(input_)
        print('Input from file', input_)
        
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
                return
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
                prev_frame = frame[roi_y: roi_y + roi_h,
                                   roi_x: roi_x + roi_w].copy()
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
    
                continue
            ttot_ = ts - tstart
            ttot = ttot_.total_seconds()
            tdelta_ = ts - tprev
            tdelta = tdelta_.total_seconds()
    
            if (duration > 0.0) and (ttot > duration):
                break
            roi = frame[roi_y: roi_y + roi_h,
                        roi_x: roi_x + roi_w]
            if motion_based:
                save = check_motion(roi, prev_frame,
                                    threshold,
                                    min_area,
                                    kernel_width)
            elif interval > 0:
                save = tdelta >= interval
            if save:
                prev_frame = roi.copy()
                tstring = ts.isoformat()
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

