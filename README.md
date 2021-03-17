# Argos: tracking multiple objects
Argos is a software utility for tracking multiple objects (animals) in a video.

## Getting started

### Installation using Anaconda
1. Install [anaconda](https://www.anaconda.com/) python distribution. You can 
   download the free Individual Edition 
   [here](https://www.anaconda.com/products/individual#Downloads)
2. Create an environment with required packages (enter this commands in 
   Anaconda prompt):
   
   ```
   conda create -n track -c conda-forge python cython scipy numpy scikit-learn pyqt pyyaml matplotlib pandas pytables ffmpeg sortedcontainers
   ```
   
   This will create a virtual Python environment called `track`
3. Activate the environment (enter this commands in Anaconda prompt):
   
   ```
   conda activate track
   ```
   
4. Install OpenCV with contributed modules (required for some recent tracking 
   algorithms, but not part of the main OpenCV distribution available in conda):
   ```commandline
    pip install opencv-contrib-python
   ```
   
5. Install PyTorch.

   If you have a CUDA capable GPU, see  [pytorch website](https://pytorch.org/get-started/locally/)
   to select the  right command. But note that you will need to install the appropriate 
   [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) for it to work.

   In case you do not have a CUDA capable GPU, you have to use
   *CPU-only* version (which can be ~10 times slower), in the Anaconda
   prompt

   ``` 
   conda install pytorch torchvision cpuonly -c pytorch 
   ``` 

6. Install `pycocotools`

   On Windows:
     1. Install [MS Visual Studio Build Tools](https://go.microsoft.com/fwlink/?LinkId=691126). 
        Select Windows XX SDK for your version of Windows.
     2. Go to `C:\Program Files (x86)\Microsoft Visual C++ Build Tools` and run 
        `vcbuildtools_msbuild.bat`
     3. Install [git](https://git-scm.com/downloads)
     4. In the Anaconda command prompt run:
	 
        ```
        pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
        ```
		If using PowerShell, use this instead:
        ```
        pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
        ```
		
		
   On Linux/Unix/Mac you need to have `make` and `g++` installed, and then in 
   the Anaconda command prompt:
   
   ```
   pip install pycocotools
   ```

7. Install Argos with this command:

   ```
   pip install argos_toolkit
   ```
   and the Tracking utility:
   ```
   pip install argos_tracker
   ```
   

### Usage

To try Argos tracking on objects in COCO dataset, download the
pretrained model released with YOLACT
[here](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing)
or go to [YOLACT repository](https://github.com/dbolya/yolact) to find
a mirror.  The corresponding configuration file is already installed
in
`{your_python_environment}/lib/site-packages/argos/config/yolact_base/yolact_base_config.yml`.

If you used Anaconda as described here, `{your_python_environment}`
should be `C:\Users\{username}\Anaconda3\env\track\` for Anaconda3 on
Windows, `~/.conda/envs/track` on Linux.

You can also download some weights and corresponding configuration
files here: [Argos example config](https://www.dropbox.com/sh/9fcgouv6wsjvypk/AAC5A2BIrbpdOG5vy8YwOk6ca?dl=0)

To run any of the utilities in the Argos toolkit, you have to switch
to the Anaconda environment created during the installation:

```
conda activate track
```

And then run the utility (except the Tracking tool) as a Python module:

```
python -m argos.{utility}
```

Continue reading below, or refer to the documentation for specific
cases.

#### Interactive tracking

The `argos_track` utility provides a graphical
interface to set the parameters, choose algorithms for detection
(instance segmentation) and tracking, and to view the performance as
tracking proceeds. Follow the steps below to start and use this
program.

9. Run `argos` tracking script on the Anaconda prompt:
   
   ```
   python -m argos_track
   ```

   This will start the Graphical User Interface for the
   tracker. Follow the steps below in the GUI to track objects while
   visualizing the tracking.

10. Open the video file using either the `File` menu. After selecting
    the video file, you will be prompted to:
	  1. Select output data directory/file. You have a choice of CSV
         (text) or HDF5 (binary) format. HDF5 is recommended.
	  2. Select Yolact configuration file, go to the `config`
         directory inside argos directory and select
         `yolact_base/yolact_base_config.yml`.
	  3. File containing trained network weights, and here you should
         select the `yolact_base_54_800000.pth` file downloaded from
         YOLACT repository page.
	  
11. Start tracking: click the `play/pause` button and you should see
    the tracked locusts. The data will be saved in the filename you
    entered in step above.

    If you choose CSV above, the bounding boxes of the segmented
    objects will be saved in `{videofile}.seg.csv` with each row
    containing `frame-no,x,y,w,h` where (x, y) is the coordinate of
    the top left corner of the bounding box and `w` and `h` are its
    width and height respectively.
    
    The tracks will be saved in `{videofile}.trk.csv`. Each row in this file 
    contains `frame-no,track-id,x,y,w,h`.
	
	If you choose HDF5 in step 10.1 above, the same data will be saved
    in a single file compatible with the Pandas library. The
    segementation data will be saved in the group `/segmented` and
    tracks will be saved in the group `/tracked`. The actual values
    are in the dataset named `table` inside each group, with columns
    in same order as described above for CSV file. You can load the
    tracks in a Pandas data frame in python with the code fragment:

	```
	tracks = pandas.read_hdf(tracked_filename, 'tracked')
	```

     
12. Classical segmentation: Using the `Segmentation method` menu you can switch
    from YOLACT to classical image segmentation for detecting target objects. 
    This method uses patterns in the pixel values in the image to detect 
    contiguous patches. If your target objects are small but have high contrast 
    with the background, this may give tighter bounding boxes, and thus more 
    accurate tracking.
    
    When this is enabled, the right panel will allow you to set the parameters.
    
    The classical segmentation works by first blurring the image so that sharp 
    edges of objects are smoothed out. Blur width and SD control this.
    
    Select invert thresholding for dark objects on light background.
    
    Segmentation method: if Threshold, then contiguous white pixels are taken 
    as objects. If DBSCAN, the DBSCAN algorithm is used for spatially 
    clustering the white pixels. This usually provides finer selection of
    object at the cost of speed. 
    
    When DBSCAN is chosen, only clusters of at least `minimum samples`	pixels 
    are considered valid.
    
    The initial segmentation is further refined by specifying limits on object 
    size by specifying `minimum pixels`, `maximum pixels`, `minimum width`, 
    `maximum width`, `minimum length` and `maximum length`.
	
	
#### Batch tracking 
You can also run the tracking in batch mode from the command
line. This is useful for processing a number of files from a shell
script. This uses YOLACT for decteting objects and SORT for tracking.

```
python -m argos_track.batchtrack -i {input_file} -o {output_file} -c {yolact_config} -w {yolact_weights} -s {score} -k {max_objects} --hmin {minimum_height} --hmax {maximum_height} --wmin {minimum_width} --wmax {maximum_width} --overlap {minimum_overlap} --max_age {maximum_misses}
```	

where every entry inside braces is to be replaced by the appropriate
value. The arguments are described below (full list can be obtained by
the command `python -m argos_track.batchtrack -h`)

- `input_file`: path of input video file

- `output_file`: path of output data

- `yolact_config`: path of yolact configuration file (as described above
  in step 10)

- `yolact_weights`: path of yolact trained weights/network file (as
  described above in step 10.3)

- `score`: a decimal fraction between 0 and 1 specifying acceptable detection
  score. 0.15 is more lenient and 0.75 is more strict. For weights
  trained to detect a single object 0.15 to 0.3 can be usable.

- `max_objects`: maximum number of object to retain. This keeps at most top k
  objects with maximum detection score.

- `minimum_height`: an integer - filter out detected objects whose
  bounding box has length in pixels less than this number.
  
- `maximum_height`: an integer - filter out detected objects whose
  bounding box has length in pixels larger than this number.
  
- `minimum_width`: an integer - filter out detected objects whose
  bounding box has width in pixels less than this number.

- `maximum_width`: an integer - filter out detected objects whose
  bounding box has width in pixels larger than this number.

- `minimum_overlap`: a decimal fraction between 0 and 1 - the minimum
  overlap between two overlapping objects in successive frames to
  consider them the same object. This overlap is measured as the ratio
  of intersection to union of their bounding boxes. Smaller value will
  be lenient, larger value will be stricter. 
  
  Imagine object A in frame 1 has moved in frame 2 to A'. If the area
  of overlap of the bounding boxes of A and A' is half their combined
  area, and the specified minimum overlap is 0.3, then A' will be
  correctly labeled the same as A. If the specified minimum overlap is
  0.7, then A' will be considered a different object and will receive
  a new label.
  
  Thus with a larger value for overlap, a small movement may cause the
  object to be labeled as a new object. A smaller value of overlap may
  cause different objects coming close together to be confused as the
  same object.
  
- maximum_misses: if an object cannot be detected in these many
  successive frames, it is considered lost. It can be smaller when
  detection is good and the video is recorded at high FPS.
  
Example:

For detecting animals that should be within 5 and 50 pixels wide and
between 10 and 100 pixels long, with the yolact configuration file in
`config/yolact_config.yml` and weights of a network trained to detect
these animals in `config/weights.pth`, the recorded video in
`myvideo.avi`, where we know that no more than 20 animals (including
misdetection of other objects as the animal, e.g. a scratch in the
arena) should be detected in the video, the following command may
work:

```
python -m argos_track.batchtrack -i myvideo.avi -o myvideo.h5 -c config/yolact.yml -w config/weights.pth -s 0.3 -k 20 --hmin 10 --hmax 100 --wmin 5 --wmax 50 --overlap 0.3 --max_age 20
```

This will give a new label to an object if it is missing for 20 frames
or more. If there are misdetections, they can be corrected manually by
the `review` tool described below.

Before embarking on processing a series of similar videos in batch
mode, it is a good idea to track a few of them in interactive mode
described earlier in order to estimate, by trial and error, the
command line arguments like minimum and maximum height and width,
overlap and maximum number of objects.


## Additional utilities
### Capture video with timestamp for each frame
- `capture.py` : a python script to record from a webcam or convert an
   existing video based on movement. For very long recordings it may
   be wasteful to record video when there is nothing happening. You
   can use this script to record or convert video so only parts where
   there is some minimum change of pixels (for example due to
   movement) are stored. Alongside the output video, it keeps a `.csv`
   file with the time of each frame. Check the source code or enter
   `python argos/capture.py -h` to find the command line arguments.

### Review tracks to manually correct mislabelings
`review.py` : a Python/Qt GUI to go through the automatically detected
tracks and correct mistakes.

1. Follow steps under `Usage` above after installation to prepare for
   running the reviewer.
2. Start the GUI using the command

  ```
  python -m argos.review
  ```

3. From the File menu open the track generated by `argos_track` and it
   will ask for the corresponding video file.
   
   Once both are selected, you will see the current frame in the right
   pane and the previous frame in the left pane (initially empty).

4. Press Play (keyboard shortcut: `space bar`) to start going through the video.

### Important items in Menu/Toolbar
- `Scroll views together` - zooming will work simultaneously on both
   left and right pane, scrolling right pane will scroll the left one
   too. Useful for comparing the same regions in a zoomed in video.
   
-  `Set color` button for selecting a single color for all bounding
   boxes and track label text.

-  `Autocolor` button when checked, will automatically pick random
   colors.
   
   `Colormap` button for selecting a colormap and number of different
   values to use from this colormap for the bounding boxes and track
   label text.

-   `Show in grayscale` will show the video in gray. Helps when the
   colors of bboxes and labels are too similar to the colors in the
   video.

- If the `Show popup message for left/right mismatch` button is
  checked (default), then it will show a popup message each time the
  track ids in the current frame do not match those on the left frame
  and the video will pause.
   
- If `Show popup message for new track` button is checked, then only
  when a new track appears on the right pane, the video will pause a
  popup message will inform you about it.
   
- If `No popup message for tracks` button is checked, then the video
  will run silently.
   
- Whenever there is a left-right mismatch in track labels, there will
  be a message in the status bar (a status message) pointing out the
  differences. In the message text, new track labels will be in bold.

- In case a track has been mislabeled, you can drag and drop the
  correct label pressing the left mouse button from the list of all
  tracks in the middle to the corresponding track id in the list of
  current tracks in the right list.
     
- To apply this for just the current frame, keep the `Shift` key
  pressed when dragging and dropping.
     
- If a track on the right is a false detection, you can delete it by
  pressing `x` or `delete` key.
     
  To apply this for current frame only keep the `Ctrl` key pressed at
  the same time.

- Sometimes the identities of two objects that are too close together
  or cross each other, can be swapped by mistake. You can use the
  right mouse button to drag and drop one track id from the left/all
  list on another on the right list to swap them.
     
  To apply this for just the current frame, keep the `Shift` key
  pressed when dragging and dropping.
     
     *NOTE* Swapping and assigning on the same trackid within a single
     frame can be problematic.  Sometimes the tracking algorithm can
     temporarily mislabel tracks. For example, object `A` (trackid=1)
     crosses over object `B` (trackid=2) and after the crossover
     object `A` got new label as trackid=3, and object `B` got
     mislabelled as trackid=1. The best order of action here is to (a)
     swap 3 and 1, and then (b) assign 2 to 3. This is because
     sometimes the label of `B` gets fixed automatically by the
     algorithm after a couple of frames. Since the swap is applied
     first, `B`'s 3 becomes 1, but there is no 1 to be switched to 3,
     thus there is no trackid 3 in the tracks list, and the assignment
     does not happen, and `A` remains 2. Had we first done the
     assignment and then the swap, `B` will get the label 2 from the
     assignment first, and as `A` also has label 2, both of them will
     become 1 after the swap.
	 
- To undo the changes made in current frame press `Ctrl+Z`. Note that
  this will undo all operations (swap, assignment, deletion)
  specified in the current frame.

- By default the reviewer only shows the current tracks on the right
  and previous frame's tracks on the left. In order to display tracks
  from past frames, check the `Show old tracks` button in the toolbar
  or the item in *View* menu.

- You can select `View->Show list of changes` (keyboard `Alt+C`) to
  display all the delete, assign and swap operations you suggested
  till the current frame. These are applied during the display of
  tracks, and when you save the data from `File->Save reviewed data`,
  the data will be saved after applying all these changes. You can
  also save the changes in a text file. This is useful if you are
  unsure of the changes you are making, and do not want to make
  permanent modifications or go through relatively slow full save of
  all track data. You can load the original track file later and load
  the change list, and these changes will be applied when you play the
  video.

- `View->Set old track age limit` will allow you to enter the number
  of past frames from which old tracks will be shown when the `Show
  old tracks` menu item is selected. This helps avoid visual clutter,
  but if too short, you will miss a track id that was lost from
  detection for longer than these many frames. The optimal number will
  depend on the quality of the original tracking, but starting with
  something between 200 and 500 may help.

- Selecting a trackid in the right track list will show the positions
  of this track label across the entire dataset in the right pane
  using a colormap with color value gradually changing with age. (TODO
  check)

- Selecting a trackid in the middle track list will show the positions
  of this track label across the entire dataset in the left pane using
  a colormap with color value gradually changing with age. (TODO
  check)

- Note that the changes like swap, assign and delete are consolidated
  only when you save the data to a file. If the tracking algorithm
  lost trackid 34 in frame 1000, and relabeled it 40 in frame 1001
  till frame 2000, even if you assigned 34 to 40, selecting 34 in the
  middle track list (all tracks) will display its positions only up to
  frame 1000 in the left pane. However, if you save the data into a
  file after the assigment, and then select 34 in the all tracks list,
  positions of track 34 will be displayed all the way to frame 2000.

- Filtering out bad tracks by ROI: if there is too much noise outside
  the area of interest in the video, and the tracker misclassified
  these as valid tracks, you can draw a polygon in the right pane over
  the video frame (after pausing the video). Click the left mouse
  button to place a vertex of the polygon, at the end click it again at
  the first vertex to complete the polygon. This will be the region of
  interest and any object whose bounding box is entirely outside this
  region of interest will be excluded.

- Filtering out bad tracks by size: if your objects of interest have a
  specific size range, you can filter out bad detections by setting a
  size limit via `View->Size limits`.
	 
- `plot_tracks.py` : a python script with functions to display the tracks.

## Known-issues
- I get the error message when switching to CSRT:
  `AttributeError: module 'cv2.cv2' has no attribute 'TrackerCSRT_create'`
  - You have installed the standard opecv / python-opencv /opencv-python package.
    For CSRT you need opencv with contribute modules. Try:
    ```
    conda remove opencv-python
    pip install opencv-contrib-python
    ```
- I get this exception when running argos after installation
  ```
  ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
  ```

  This happens with older version of numpy (should be 1.20 or later). 
  Try upgrading numpy:
  ```
  pip install -U numpy
  ```

- When installing with `pip` I get this error message 
  ```
  Collecting torch
    Downloading torch-1.8.0-cp39-cp39-manylinux1_x86_64.whl (735.5 MB)
  
  ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE. If you have updated the package versions, please update the hashes. Otherwise, examine the package contents carefully; someone may have tampered with them.
  ```
  Try pip with `--no-cache-dir` option, like this:
  ```
  pip install --no-cache-dir torch
  ```
- I get this exception when trying to run argos tracker
  ```
  ModuleNotFoundError: No module named 'pycocotools'
  ```
  
  This indicates that pycocotools is not installed on your system. We
  did not include pycocotools in the dependencies as that creates
  problem for MS Windows (see special case for Windows in installation
  instrctions above).
  
  On Unix-like systems (Linux/Mac) you can install pycocotools with
  ```
  pip install pycocotools
  ```
  
- I get this error when trying `python -m argos_track`:
  ```
  RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx
  ```
  
  Check if you have NVIDIA drivers for CUDA installed. Also note that
  CUDA does not work from Windows Subsystem for Linux (WSL). In
  general it is a good idea to install Argos on the native platform.

## Credits

- [SORT](https://github.com/abewley/sort) was developed by Alex
  Bewley. The related publication is:
  ```
  @inproceedings{Bewley2016_sort,
    author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
    booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
    title={Simple online and realtime tracking},
    year={2016},
    pages={3464-3468},
    keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
    doi={10.1109/ICIP.2016.7533003}
  }
  ```
  
- [YOLACT](https://github.com/dbolya/yolact) was developed by Daniel
  Bolya. The related publication is:
  ```
  @inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
  }
  ```
