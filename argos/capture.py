# capture.py ---
# Author: Subhasis Ray
# Created: Thu Feb 27 17:31:02 2020 (-0500)
# Code:


"""
========================
Capture or process video
========================
Usage:
::
    python -m argos.capture -i 0 -o myvideo_motion_cap.avi

To see a list of all options try ``python -m argos.capture -h``

The above command will take a snapshot with the default camera on your
computer and ask you to select the region of interest. Click your left
mouse button on one corner of the region of interest and drag to draw
a rectangle. Press ``Enter`` key when done (you may need to press it
twice). If you want to change the selection, or moved the camera to
adjust focus or frame, press ``C`` on the keyboard to update the image
and draw the ROI again. Press ``Enter`` to start recording. This will
create two files, ``myvideo_motion_cap.avi`` containing the recorded
video and ``myvideo_motion_cap.avi.csv`` with the timestamp of each
frame.

`argos.capture` is a simple tool to capture video along with timestamp
for each frame using a camera. It can also be used for recording
videos only when some movement is detected. When applied to a
pre-recorded video file, enabling motion-based capture will keep only
those frames between which *significant movement* has been detected.

What is significant movement?

- The movement detection works by first converting the image into
  gray-scale and blurring it to make it smooth. The size of the
  Gaussian kernel used for blurring is specified by the
  ``--kernel_width`` parameter.

- Next, this blurred grayscale image is thresholded with threshold
  value specified by the ``--threshold`` parameter.

- The resulting binary frame is compared with the blurred and
  thresholded version of the last saved frame. If there is any patch
  of change bigger than ``--min_area`` pixels, then this is considered
  significant motion.

Not all video formats are available on all platforms. The default is
MJPG with AVI as container, which is supported natively by OpenCV.

If you need high compression, X264 is a good option. Saving in X264
format requires H.264 library, which can be installed as follows:

- On Linux: ``conda install -c anaconda openh264``

- On Windows: download released binary from here:
  https://github.com/cisco/openh264/releases and save them in your
  library path.

Examples
--------
1. Read video from file ``myvideo.mpg`` and save output in
``myvideo_motion_cap.avi`` in DIVX format. The ``-m --threshold=20 -a
10`` part tells the program to detect any movement such that more than
10 contiguous pixels have changed in the frame thresholded at 20:
frames
::
    python -m argos.capture -i myvideo.mpg -o myvideo_motion_cap.avi  \\
    --format=DIVX -m --threshold=20 -a 10

The recording will stop when the user presses ``Escape`` or ``Q`` key.

2. Record from camera# 0 into the file ``myvideo_motion_cap.avi``:
::
    python -m argos.capture -i 0 -o myvideo_motion_cap.avi

3. Record from camera# 0 for 24 hours, saving every 10,000 frames into a
separate file:
::
    python -m argos.capture -i 0 -o myvideo_motion_cap.avi \\
    --duration=24:00:00 --max_frames=10000

This will produce files with names ``myvideo_motion_cap_000.avi``,
``myvideo_motion_cap_001.avi``, etc. along with the timestamps in files named
``myvideo_motion_cap_000.avi.csv``, ``myvideo_motion_cap_001.avi.csv``. The
user can stop recording at any time by pressing ``Escape`` or ``Q`` key.

4. Record a frame every 3 seconds:
::
    python -m argos.capture -i 0 -o myvideo_motion_cap.avi --interval=3.0

Common problem
--------------

When trying to use H264 format for saving video, you may see the
following error:
::
    Creating output file video_filename.mp4
    OpenCV: FFMPEG: tag 0x34363248/'H264' is not supported with codec id 27 and
    format \'mp4 / MP4 (MPEG-4 Part 14)\'
    OpenCV: FFMPEG: fallback to use tag 0x31637661/\'avc1\'

            OpenH264 Video Codec provided by Cisco Systems, Inc.


Solution: Use .avi instead of .mp4 extension when specifying output filename.

"""

import sys
import os
import argparse
import time
from datetime import datetime, timedelta
import csv
import cv2
import numpy as np
from matplotlib import colors

from argos import caputil

has_ccapture = False

try:
    from argos.ccapture import vcapture as cvcapture
    has_ccapture = True
    print('Loaded C-function for video capture.')
except ImportError as err:
    print('Could not load C-function for video capture.'
          ' Using pure Python.')
    print(err)


def check_motion(current, blurred_prev, blurred_cur,
                 threshold, min_area, kernel_width=21,
                 show_diff=False):
    """As a side effect, save the current blurred image in blurred_prev to
    reduce memory allocation overhead and match cython
    implementation

    """
    gray_cur = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    blurred_cur[:, :] = cv2.GaussianBlur(gray_cur, (kernel_width,
                                                    kernel_width), 0)
    frame_delta = cv2.absdiff(blurred_cur, blurred_prev)
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
    if len(moving_contours) > 0:
        blurred_prev[:, :] = blurred_cur
    return len(moving_contours) > 0


def vcapture(params):
    try:
        input_ = int(params['input'])
    except ValueError:
        input_ = params['input']
    win_name = f'Input: {input_}'
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
    blurred_prev = None
    blurred = None
    if isinstance(input_, int):
        tstart = datetime.now()
        if params['width'] > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, params['width'])
        if params['height'] > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, params['height'])
        print(f'Suggested size: {params["width"]}x{params["height"]}')
        print(f'Actual size: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x'
              f'{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
        if params['config']:
            cap.set(cv2.CAP_PROP_SETUP, 1)
    else:
        tstart = datetime.fromtimestamp(time.mktime(time.localtime(
            os.path.getmtime(input_))))
    ret, frame = cap.read()
    if frame is None:
        raise Exception('Could not get frame')
    if not isinstance(input_, int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    save = True
    t_prev = None
    contours = []
    file_count = 0
    read_frames = 0
    writ_frames = 0
    roi_x = params['roi_x']
    roi_y = params['roi_y']
    roi_w = params['roi_w']
    roi_h = params['roi_h']
    try:
        while True:
            ret, frame = cap.read()
            if frame is None:
                print('Empty frame')
                break
            frame = frame[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w].copy()
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
                output_file = f'{fname}_{file_count:03d}.{ext}' \
                    if params['max_frames'] > 0 else params['output']
                while os.path.exists(output_file):
                    file_count += 1
                    output_file = f'{fname}_{file_count:03d}.{ext}'
                file_count += 1
                timestamp_file = f'{output_file}.csv'
                print('Creating output file', output_file)
                t_prev = ts
                # if isinstance(input_, int):
                #     tstart = ts
                print(f'Original frame shape: '
                      f'Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} '
                      f'Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}\n'
                      f'ROI width: {width}, height: {height}')

                fourcc = cv2.VideoWriter_fourcc(*params['format'])
                out = cv2.VideoWriter(output_file, fourcc, params['fps'],
                                      (roi_w, roi_h))
                tsout = open(timestamp_file, 'w', newline='')
                tswriter = csv.writer(tsout)
                tswriter.writerow(['inframe', 'outframe', 'timestamp'])
                writ_frames = 1

            # Check if current time is more than specified duration since start.
            # If so, stop recording.
            time_from_start = ts - tstart
            time_from_start = time_from_start.total_seconds()
            time_delta = ts - t_prev
            time_delta = time_delta.total_seconds()
            # print(f'Time from start {time_from_start}, specified duration {params["duration"]} s')
            if (params['duration'] > 0) and (
                    time_from_start > params['duration']):
                break

            if params['motion_based']:
                if blurred is None:
                    blurred = np.zeros(frame.shape[:2], dtype=np.uint8)
                    blurred_prev = np.zeros(frame.shape[:2], dtype=np.uint8)
                save = check_motion(frame, blurred_prev, blurred,
                                    params['threshold'],
                                    params['min_area'],
                                    kernel_width=params['kernel_width'],
                                    show_diff=params['show_diff'])
            elif params['interval'] > 0:
                save = time_delta >= params['interval']

            if save:
                tstring = ts.isoformat()
                if params['timestamp']:
                    if params['tb'] is not None:
                        (w, h), baseline = cv2.getTextSize(
                            tstring, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (params['tx'], params['ty']),
                                      (params['tx'] + w, params['ty'] - h),
                                      params['tb'], cv2.FILLED)
                    cv2.putText(frame, tstring,
                                (params['tx'], params['ty']),
                                cv2.FONT_HERSHEY_SIMPLEX, params['fs'],
                                params['tc'], 2, cv2.LINE_AA)
                out.write(frame)
                tswriter.writerow(
                    [read_frames - 1, writ_frames - 1, tstring])
                # logger.debug(f'{read_frames}\t{writ_frames}\t{tstring}')
                writ_frames = ((writ_frames + 1) % params['max_frames']) \
                    if params['max_frames'] > 0 else (writ_frames + 1)
                t_prev = ts
                if params['interactive']:
                    cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            # if `q` or ESCAPE is pressed, break from the loop
            if key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        print('Caught keyboard interrupt')
    finally:
        print('Closing')
        cap.release()
        if out is not None:
            out.release()
        if tsout is not None:
            tsout.close()
        cv2.destroyAllWindows()
    return read_frames


def parse_interval(tstr):
    """Convert string for time interval into `timedelta`.

    Parameters
    ----------
    tstrt: str
        String representing time interval. Should be in `HH:MM:SS` format.
        The seconds part can have fractional part.
    Returns
    -------
        datetime.timedelta
    """
    h = 0
    m = 0
    s = 0
    values = tstr.split(':')
    if len(values) < 1:
        raise ValueError('At least the number of seconds must be specified')
    elif len(values) == 1:
        s = float(values[0])
    elif len(values) == 2:
        m, s = (int(values[0]), float(values[1]))
    elif len(values) == 3:
        h, m, s = (int(values[0]), int(values[1]), float(values[2]))
    else:
        raise ValueError('Expected duration in hours:minutes:seconds format')
    t = timedelta(hours=h, minutes=m, seconds=s)
    return t


def check_params(args):
    """Check/cleanup parameter values in `args` and convert to a `dict`.

    """
    params = {}
    params.update(args)
    input_ = params['input']
    camera = False
    try:
        input_ = int(input_)
        camera = True
    except ValueError:
        assert len(input_.strip()) > 0, \
            'Input must be a number or path to a video file'

    if camera:
        fps, width, height = caputil.get_camera_fps(
            input_,
            params['width'],
            params['height'], fps=30, nframes=30)
    else:
        fps, width, height = caputil.get_video_fps(input_)
    print(f'Detected effective input FPS = {fps}')
    if params['fps'] <= 0:
        params['fps'] = fps
    if params['width'] <= 0 or params['height'] <= 0:
        params['width'] = width
        params['height'] = height
    if len(params['output'].strip()) == 0:
        # Windows forbids ':' in filename
        ts = datetime.now().isoformat().replace(':', '')
        params['output'] = f'video_{ts}.avi'
    duration = params['duration']
    if len(duration) > 0:
        duration = parse_interval(duration)
        params['duration'] = duration.total_seconds()
    else:
        params['duration'] = -1
    print(f'Duration {duration}, {params["duration"]}')
    r, g, b = [int(val) * 255 for val in colors.to_rgb(params['tc'])]
    params['tc'] = (b, g, r)
    if len(params['tb']) > 0:
        r, g, b = [int(val) * 255 for val in colors.to_rgb(params['tb'])]
        params['tb'] = (b, g, r)
    else:
        params['tb'] = (0, 0, 0)

    return params


def make_parser():
    parser = argparse.ArgumentParser('Record video based on motion detection.'
                                     ' It can create an output video file with'
                                     ' only the frames between which some'
                                     ' motion was detected.'
                                     ' It also dumps a csv file with the'
                                     ' input frame no., output frame no., and'
                                     ' the time stamp for the output frame in'
                                     ' the input file.')
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
    motion_group.add_argument('--threshold', type=int, default=10,
                              help='Grayscale threshold value for detecting'
                                   ' change in pixel value between frames')
    motion_group.add_argument('-a', '--min_area', type=int, default=100,
                              help='Area in pixels that must change in'
                                   ' order to consider it actual movement as'
                                   'opposed to noise.'
                                   ' Works with --motion_based option')
    # motion_group.add_argument('--show_contours', action='store_true',
    #                           help='Draw the contours exceeding `min_area`.'
    #                                ' Useful for debugging')
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
    parser.add_argument('--interval', type=float, default=-1,  #
                        help='Interval in seconds between acquiring frames.')
    parser.add_argument('--duration', type=str, default='',
                        help='Duration of recordings in HH:MM:SS format.'
                             ' If unspecified or empty string, we will record'
                             'indefinitely.')
    parser.add_argument('--interactive', type=int, default=1,
                        help='Whether to display video as it gets captured. '
                             'Setting it to 0 may speed up things a bit.')
    parser.add_argument('--roi', type=int, default=1,
                        help='Whether to select ROI.')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='After these many frames, save in a '
                             'new video file')
    parser.add_argument('--format', type=str, default='MJPG',
                        help='Output video codec, see '
                             'http://www.fourcc.org/codecs.php for description'
                             ' of available codecs on different platforms.'
                             ' default X264 produces the smallest videos')
    camera_group = parser.add_argument_group(
        title='Camera settings',
        description='Parameters to set camera properties')
    camera_group.add_argument('--fps', type=float, default=30,
                              help='frames per second, if recording from a'
                                   ' camera, and set to negative, we '
                                   'record 120 frames first to estimate frame'
                                   ' rate of camera.')
    camera_group.add_argument('--width', type=int, default=-1,
                              help='Frame width')
    camera_group.add_argument('--height', type=int, default=-1,
                              help='Frame height')
    parser.add_argument('--config', action='store_true',
                        help='Show dialog to configure camera.')
    parser.add_argument('--use_c', action='store_true',
                        help='Try to use Cython implementation')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    params = check_params(vars(args))
    width, height = params['width'], params['height']
    print('Width', width, 'Height', height)
    if params['roi'] == 0:
        roi_x, roi_y, roi_w, roi_h = 0, 0, width, height
    else:
        roi_x, roi_y, roi_w, roi_h, width, height = caputil.get_roi(
            params['input'], params['width'], params['height'])

    params.update({'roi_x': roi_x,
                   'roi_y': roi_y,
                   'roi_w': roi_w,
                   'roi_h': roi_h,
                   'width': width,
                   'height': height})
    read_frames = 0
    start = time.perf_counter()
    if has_ccapture and params['use_c']:
        read_frames = cvcapture(
            params['input'], params['output'], params['format'],
            params['fps'],
            params['interactive'],
            params['width'], params['height'],
            params['roi_x'], params['roi_y'],
            params['roi_w'], params['roi_h'],
            params['interval'],
            params['duration'],
            params['max_frames'],
            params['motion_based'],
            params['config'],
            params['threshold'],
            params['min_area'],
            params['kernel_width'],
            params['timestamp'],                 
            params['show_diff'],
            np.array(params['tb'], dtype=int),
            np.array(params['tc'], dtype=int),
            params['tx'],
            params['ty'],
            params['fs']
        )
    else:
        read_frames = vcapture(params)
    end = time.perf_counter()
    dt = end - start
    print(f'Read {read_frames} frames in {dt} seconds')

#
# capture.py ends here
