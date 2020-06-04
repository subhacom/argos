# capture.py ---
# Author: Subhasis Ray
# Created: Thu Feb 27 17:31:02 2020 (-0500)
# Code:


"""Program to capture video using OpenCV.

We allow motion based recording: only when there is significant change
in scene.

Each frame can be marked with timestamp. It will also create a logfile
with timestamps for each frame.

This uses OpenCV 3: `conda install opecv-python`

Saving in X264 format requires H.264 library:

Linux: `conda install -c anaconda openh264`

Windows: download released binary from here:
https://github.com/cisco/openh264/releases and save them in your
library path (could be this directory).

Error:
```
Creating output file video_2020_02_28__12_52_33.mp4
OpenCV: FFMPEG: tag 0x34363248/'H264' is not supported with codec id 27 and
format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x31637661/'avc1'

        OpenH264 Video Codec provided by Cisco Systems, Inc.
```

Solution: Use .avi instead of .mp4 extension

"""


import sys
import os
import argparse
import time
from datetime import datetime, timedelta
import csv
import cv2
# import logging
# from logging.handlers import MemoryHandler
from matplotlib import colors


CV2_MAJOR, CV2_MINOR, _ = cv2.__version__.split(".")
CV2_MAJOR = int(CV2_MAJOR)
CV2_MINOR = int(CV2_MINOR)

# logger = logging.getLogger('frame_info')

#
# def setup_logger(fname, buffered=False):
#     global logger
#     logger.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s\t%(message)s')
#     file_handler = logging.FileHandler(fname, mode='w')
#     file_handler.setFormatter(formatter)
#     # file_handler.setLevel(logging.INFO)
#     if buffered:
#         mem_handler = MemoryHandler(capacity=1000,
#                                     flushLevel=logging.ERROR,
#                                     target=file_handler)
#         # mem_handler.setFormatter(formatter)
#         logger.addHandler(mem_handler)
#     else:
#         logger.addHandler(file_handler)


def make_parser():
    parser = argparse.ArgumentParser('Record video based on motion detection')
    parser.add_argument('-i', '--input', type=str, default='0',
                        help='Input source, '
                        'if unspecified or 0 use first available camera;'
                        'if string, extract motion from video file;'
                        'if an integer, that camera number.')
    parser.add_argument('-o', '--output', type=str, default='',
                        help='output file path')
    motion_group = parser.add_argument_group(
        title='Motion based',
        description='Capture video based on motion')
    motion_group.add_argument('-m', '--motion_based', action='store_true',
                              help='Whether to use motion detection')
    motion_group.add_argument('-k', '--kernel_width', type=int, default=21,
                              help='Width of the Gaussian kernel for smoothing'
                              ' the frame before comparison with previous'
                              'frame. Small changes are better detected '
                              ' With a smaller value (21 works well for about'
                              ' 40x10 pixel insects)')
    motion_group.add_argument('--threshold', type=int, default=100,
                              help='Grayscale threshold value for detecting'
                              ' change in pixel value between frames')
    motion_group.add_argument('-a', '--min_area', type=int, default=100,
                              help='Area in pixels that must change in'
                              ' order to consider it actual movement as'
                              'opposed to noise.'
                              ' Works with --motion_based option')
    motion_group.add_argument('--show_contours', action='store_true',
                              help='Draw the contours exceeding `min_area`.'
                              ' Useful for debugging')
    motion_group.add_argument('--show_diff', action='store_true',
                              help='Show the absolute difference between'
                              ' successive frames and the thresholded '
                              ' difference in two additional windows.'
                              ' Useful for debugging and choosing parameters'
                              ' for motion detection.'
                              'NOTE: the diff values are displayed using'
                              ' the infamous jet colormap, which turns out'
                              ' to be good at highlighting small differences')
    timestamp_group = parser.add_argument_group(
        title='Timestamp parameters',
        description='Parameters to display timestamp in recorded frame')
    timestamp_group.add_argument('-t', '--timestamp', action='store_true',
                                 help='Put a timestamp each recorded frame')
    timestamp_group.add_argument('--tx', type=int, default=15,
                                 help='X position of timestamp text')
    timestamp_group.add_argument('--ty', type=int, default=15,
                                 help='Y position of timestamp text')
    timestamp_group.add_argument('--tc', type=str, default='#ff0000',
                                 help='Color of timestamp text in web format'
                                 ' (#RRGGBB)')
    timestamp_group.add_argument('--tb', type=str, default='',
                                 help='Background color for timestamp text')
    timestamp_group.add_argument('--fs', type=float, default=1.0,
                                 help='Font scale for timestamp text '
                                 '(this is not font size).')
    parser.add_argument('--interval', type=float, default=0.033,  # 30 FPS
                        help='Interval in seconds between acquiring frames.')
    parser.add_argument('--duration', type=str, default='',
                        help='Duration of recordings in HH:MM:SS format.'
                        ' If unspecified or empty string, we will record'
                        'indefinitely.')
    parser.add_argument('--interactive', action='store_true',
                        help='Whether to display video as it gets captured')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='After these many frames, save in a '
                        'new video file')
    parser.add_argument('--format', type=str, default='H264',
                        help='Output cideo codec, see '
                        'http://www.fourcc.org/codecs.php for description'
                        ' of available codecs on different platforms.'
                        ' default X264 produces the smallest videos')
    camera_group = parser.add_argument_group(
        title='Camera settings',
        description='Parameters to set camera properties')
    camera_group.add_argument('--fps', type=float, default=-1,
                              help='frames per second, if recording from a'
                              ' camera, and left unspecified (or -1), we '
                              'record 120 frames first to estimate frame'
                              ' rate of camera.')
    camera_group.add_argument('--width', type=int, default=-1,
                              help='Frame width')
    camera_group.add_argument('--height', type=int, default=-1,
                              help='Frame height')
    # parser.add_argument('--logbuf', action='store_true',
    #                     help='Buffer the log messages to improve speed')
    return parser


def check_motion(current, prev, threshold, min_area, kernel_width=21,
                 show_diff=False):
    gray_cur = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.GaussianBlur(gray_cur, (kernel_width,
                                           kernel_width), 0)
    gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.GaussianBlur(gray_prev, (kernel_width,
                                             kernel_width), 0)
    frame_delta = cv2.absdiff(gray_cur, gray_prev)
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
    # Useful for debugging and for exploring motion detection parameters
    if show_diff:
        win_diff_name = 'DEBUG - absdiff between frames'
        cv2.namedWindow(win_diff_name, cv2.WINDOW_NORMAL)
        win_thresh_name = 'DEBUG - thresholded absdiff'
        cv2.namedWindow(win_thresh_name, cv2.WINDOW_NORMAL)
        frame = cv2.applyColorMap(frame_delta, cv2.COLORMAP_JET)
        cv2.drawContours(frame, contours, -1, -1)
        cv2.imshow(win_diff_name, frame)
        cv2.imshow(win_thresh_name, thresh_img)
    moving_contours = [contour for contour in contours
                       if cv2.contourArea(contour) > min_area]
    return len(moving_contours) > 0, moving_contours


def capture(params):
    input_ = params['input']
    win_name = 'Video'
    if params['interactive']:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    print(f'Opening input: {input_} of type '
          f'{"DEVICE" if isinstance(input_, int) else "VIDEOFILE"}')
    if (sys.platform == 'win32') and isinstance(input_, int):
        cap = cv2.VideoCapture(input_, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(input_)
    assert cap.isOpened(), f'Could not open input {input_}'
    out = None
    tsout = None
    tswriter = None
    timestamp_file = None
    prev_frame = None
    if isinstance(input_, int):
        tstart = None
        if params['width'] > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, params['width'])
        if params['height'] > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, params['height'])
    else:
        tstart = datetime.fromtimestamp(time.mktime(time.localtime(
            os.path.getmtime(input_))))
    ret, frame = cap.read()
    if frame is None:
        raise Exception('Could not get frame')
    if not isinstance(input_, int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    x, y, w, h = cv2.selectROI('Select ROI. Press ENTER when done,'
                               ' C to continue with default.', frame)
    cv2.destroyAllWindows()
    print('ROI', x, y, w, h)
    if w == 0 or h == 0:
        x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]

    save = True
    t_prev = None
    contours = []
    file_count = 0
    read_frames = 0
    writ_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if frame is None:
                print('Empty frame')
                break
            frame = frame[y: y + h, x: x + w].copy()
            # For camera capture, use system timestamp.  For exisiting
            # video file, add frame delay to file modification timestamp.
            if isinstance(input_, int):
                ts = datetime.now()
            else:
                ts = tstart + timedelta(days=0,
                                        seconds=read_frames / params['fps'])
            read_frames += 1

            if writ_frames == 0:
                fname, _, ext = params['output'].rpartition('.')
                output_file = f'{fname}_{file_count}.{ext}' \
                    if params['max_frames'] > 0 else params['output']
                file_count += 1
                timestamp_file = f'{output_file}.csv'
                print('Creating output file', output_file)
                t_prev = ts
                prev_frame = frame.copy()
                if isinstance(input_, int):
                    tstart = ts
                print(f'Frame shape: {frame.shape}\n'
                      f'Width: {cap.get(3)} Height: {cap.get(4)}')
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*params['format'])
                out = cv2.VideoWriter(params['output'], fourcc, params['fps'],
                                      (width, height))
                # logger.debug(f'input_frame\toutput_frame\ttimestamp')
                # logger.debug(f'{read_frames}\t{writ_frames}\t{ts.isoformat()}')
                tsout = open(timestamp_file, 'w', newline='')
                tswriter = csv.writer(tsout)
                tswriter.writerow(['frame', 'timestamp'])
                writ_frames = 1
                continue

            # Check if current time is more than specified duration since start.
            # If so, stop recording.
            time_from_start = ts - tstart
            time_from_start = time_from_start.seconds + \
                time_from_start.microseconds * 1e-6
            time_delta = ts - t_prev
            time_delta = time_delta.seconds + time_delta.microseconds * 1e-6
            if (params['duration'] > 0) and (time_from_start > params['duration']):
                break

            if params['motion_based']:
                save, contours = check_motion(frame, prev_frame,
                                              params['threshold'],
                                              params['min_area'],
                                              kernel_width=params['kernel_width'],
                                              show_diff=params['show_diff'])
            elif params['interval'] > 0:
                save = time_delta >= params['interval']

            if save:
                prev_frame = frame.copy()
                tstring = ts.isoformat()
                if params['timestamp']:
                    if params['tb'] is not None:
                        (w, h), baseline = cv2.getTextSize(
                            tstring, cv2.FONT_HERSHEY_SIMPLEX, 0.5,  2)
                        cv2.rectangle(frame, (params['tx'], params['ty']),
                                      (params['tx'] + w, params['ty'] - h),
                                      params['tb'], cv2.FILLED)
                    cv2.putText(frame, tstring, (params['tx'], params['ty']),
                                cv2.FONT_HERSHEY_SIMPLEX, params['fs'],
                                params['tc'], 2)
                if params['show_contours'] and (len(contours) > 0):
                    print('Color:', params['tc'])
                    cv2.drawContours(frame, contours, -1, params['tc'], 2)
                # TODO In production writing should happen before drawing contours
                #      Displaying contours is good for debugging
                out.write(frame[y: y + h, x: x + w])
                tswriter.writerow([writ_frames, ts])
                # logger.debug(f'{read_frames}\t{writ_frames}\t{tstring}')
                writ_frames = ((writ_frames + 1) % params['max_frames']) \
                    if params['max_frames'] > 0 else (writ_frames + 1)
                t_prev = ts
            if params['interactive']:
                cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key is pressed, break from the lop
            if key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        print('Caught keyboard interrupt')
    finally:
        print('Closing')
        cap.release()
        if out is not None:
            out.release()
        tsout.close()
        cv2.destroyAllWindows()


def get_camera_fps(devid, nframes=120):
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(devid, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(devid)
    start = datetime.now()
    assert cap.isOpened(), 'Could not open camera'
    for ii in range(nframes):
        ret, frame = cap.read()
    cap.release()
    end = datetime.now()
    delta = end - start
    interval = delta.seconds + delta.microseconds * 1e-6
    fps = nframes / interval
    return fps


def get_video_fps(fname):
    cap = cv2.VideoCapture(fname)
    assert cap.isOpened(), f'Could not open video file {fname}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def check_params(args):
    params = {}
    params.update(args)
    input_ = params['input']
    try:
        input_ = int(input_)
        params['input'] = input_
    except ValueError:
        assert len(input_.strip()) > 0, \
            'Input must be a number or path to a video file'
    camera = isinstance(input_, int)  # recording from camera
    if params['fps'] <= 0:
        if camera:
            params['fps'] = get_camera_fps(input_)
        else:
            params['fps'] = get_video_fps(input_)
        print(f'Found FPS = {params["fps"]}')
    if len(params['output'].strip()) == 0:
        # Windows forbids ':' in filename
        ts = datetime.now().isoformat().replace(':', '')
        params['output'] = f'video_{ts}.avi'
    duration = params['duration']
    if len(duration) > 0:
        duration = datetime.strptime('%H:%M:%S')
        params['duration'] = duration.seconds + duration.microseconds * 1e-6
    else:
        params['duration'] = -1
    r, g, b = [int(val) * 255 for val in colors.to_rgb(params['tc'])]
    params['tc'] = (b, g, r)
    if len(params['tb']) > 0:
        r, g, b = [int(val) * 255 for val in colors.to_rgb(params['tb'])]
        params['tb'] = (b, g, r)
    else:
        params['tb'] = None

    return params


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    params = check_params(vars(args))
    # setup_logger(f'{params["output"]}.log', params['logbuf'])
    capture(params)


#
# capture.py ends here
