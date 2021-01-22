# caputil.py ---
#
# Filename: caputil.py
# Description:
# Author: Subhasis Ray
# Created: Tue Jan 12 23:21:16 2021 (-0500)
# Last-Updated: Fri Jan 22 17:00:35 2021 (-0500)
#           By: Subhasis Ray
#

# Code:

"""
==========================================================
Common utility functions and constants for capturing video
==========================================================
"""

import sys
from datetime import datetime
import cv2


CV2_MAJOR, CV2_MINOR, _ = cv2.__version__.split(".")
CV2_MAJOR = int(CV2_MAJOR)
CV2_MINOR = int(CV2_MINOR)
LARGE_FRAME_SIZE = 10000


def get_roi(input_, width, height):
    """Select region of interest (ROI) to actually save in video.

    This function opens video or camera specified by `input_` and
    shows the first frame for ROI selection. The user can left click
    and drag the mouse to select a rectangular area. Pressing Enter
    accepts the ROI selection, C cancels it. This function keeps
    updating the frame until a valid ROI is accepted.

    Parameters
    ----------
    input_: str
        input video or device (a numeric string is considered camera
        number, a file path as a video file.
    width: int
        suggested frame width for frame capture from camera
    height: int
        suggested frame height for frame capture from camera

    Returns
    -------
    tuple
        (x, y, w, h, width, height) where the first four specify ROI,
        the last two actual frame width and height.
    """
    roi = 'Select ROI. Press ENTER when done, C to try again'
    cv2.namedWindow(roi, cv2.WINDOW_NORMAL)
    x, y, w, h = 0, 0, -1, -1
    try:
        input_ = int(input_)
    except ValueError:
        pass
    capture = cv2.VideoCapture(input_)
    if not capture.isOpened():
        print(f'Could not open video {input_}')
        return None
    if isinstance(input_, int):
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
    """Find the approximate frame rate of camera.

    Try to capture some frames at specified size and frame rate, and
    calculate the actual frame rate from the time taken.

    Parameters
    ----------
    devid: int
        Camera number.
    width: int
        Suggested frame width.
    height: int
        Suggested frame height.
    fps: float
        Suggested frame rate (frames per second).
    nframes: int
        Number of frames to record for estimating average frame rate.
    Returns
    -------
    fps: float
        Estimated frame rate of the camera.
    width: int
        Actual frame width.
    height: int
        Actual frame height.
    """
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(devid, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(devid)
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
    start = datetime.now()
    for ii in range(nframes):
        ret, frame = cap.read()
    end = datetime.now()
    delta = end - start
    fps = nframes / delta.total_seconds()
    cap.release()
    return fps, width, height


def get_video_fps(fname):
    """Retrive the frame rate of video in file `fname`

    Parameters
    ----------
    fname: str
        File path.
    Returns
    -------
    fps: float
        Frame rate of video (frames per second).
    width: int
        Actual frame width.
    height: int
        Actual frame height.
    """
    cap = cv2.VideoCapture(fname)
    assert cap.isOpened(), f'Could not open video file {fname}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return fps, int(width), int(height)

#
# caputil.py ends here
