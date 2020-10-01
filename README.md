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
		
   On Linux/Unix/Mac you need to have `make` and `g++` installed, and then in 
   the Anaconda command prompt:
   
   ```
   pip install pycocotools
   ```
### Usage
7. In the Anaconda prompt, go to where `argos` is unpacked:

   ```
   cd {your_argos_directory} 
   ```
   
   it should have these directories there: `argos`, `config`, and `yolact`.
8. In the Anaconda prompt, update Python path to include this directory:

   on Windows command prompt:
   
   ```
   set PYTHONPATH=.;$PYTHONPATH
   ```
   
   and on Linux/Unix/Mac with `bash` shell:
   
   ```
   export PYTHONPATH=.:$PYTHONPATH
   ```
9. Run `argos` main script on the Anaconda prompt:

   on Windows: 
   
   
   ```
   python argos\amain.py
   ```
   
   and on Linux/Unix/Mac:
   
   ```
   python argos/amain.py
   ```
10. Open the video file using either the `File` menu. After selecting the video
   file, you will be prompted to:
   1. Select output data directory. 
   2. Select Yolact configuration file, go to the `config` directory inside 
      argos directory and select `yolact.yml`.
   3. File containing trained network weights, and here you should select the 
      `babylocust_resnet101_119999_240000.pth` file.
11. Start tracking: click the `play/pause` button and you should see the 
    tracked locusts. The data will be saved in the directory you chose in step 
    above.

    The bounding boxes of the segmented objects will be saved in 
    `{videofile}.seg.csv` with each row containing `frame-no,x,y,w,h` where 
    (x, y) is the coordinate of the top left corner of the bounding box and 
    `w` and `h` are its width and height respectively.
    
    The tracks will be saved in `{videofile}.trk.csv`. Each row in this file 
    contains `frame-no,track-id,x,y,w,h`.
     
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

## Additional utilities
### Capture video with timestamp for each frame
- `capture.py` : a python script to record from a webcam or convert an existing 
   video based on movement. For very long recordings it may be wasteful to 
   record video when there is nothing happening. You can use this script to 
   record or convert video so only parts where there is some minimum change of 
   pixels (for example due to movement) are stored. Alongside the output video, 
   it keeps a `.csv` file with the time of each frame. Check the source code or 
   enter `python argos/capture.py -h` to find the command line arguments.

### Review tracks to manually correct mislabelings
`review.py` : a Python/Qt GUI to go through the automatically detected
tracks and correct mistakes.

- TODO: Complete its own user-guide.

- From the File menu open the track generated by `argos.track` and it
  will ask for the corresponding video file.

- Once both are selected, you will see the current frame in the right
  pane and the previous frame in the left pane (initially empty).

- Press Play (keyboard shortcut: `space bar`) to start going through the video.

- `Scroll views together` - zooming will work simultaneously on both
   left and right pane, scrolling right pane will scroll the left one
   too. Useful for comparing the same regions in a zoomed in video.
   
-  `Set color` button for selecting a single color for all bounding boxes and track label text.

-  `Autocolor` button when checked, will automatically pick random colors.
   
   `Colormap` button for selecting a colormap and number of different
   values to use from this colormap for the bounding boxes and track
   label text.

-   `Show in grayscale` will show the video in gray. Helps when the
   colors of bboxes and labels are too similar to the colors in the
   video.

-   If the `Show popup message for left/right mismatch` button is checked (default), then
   it will show a popup message each time the track ids in the current
   frame do not match those on the left frame and the video will pause.
   
-   If `Show popup message for new track` button is checked, then only
   when a new track appears on the right pane, the video will pause a
   popup message will inform you about it.
   
-   If `No popup message for tracks` button is checked, then the video will run silently.
   
-   Whenever there is a left-right mismatch in track labels, there will
   be a message in the status bar (a status message) pointing out the
   differences. In the message text, new track labels will be in bold.

- In case a track has been mislabeled, you can drag and drop the
     correct label pressing the left mouse button from the list of all tracks in
     the middle to the corresponding track id in the list of current tracks in 
     the right list.
     
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

- By default the reviewer only shows the current tracks on the
     right and previous frame's tracks on the left. In order to
     display tracks from past frames, check the `Show old tracks`
     button in the toolbar or the item in *View* menu.

- You can select `View->Show list of changes` (keyboard `Alt+C`) to
     display all the delete, assign and swap operations you suggested
     till the current frame. These are applied during the display of
     tracks, and when you save the data from `File->Save reviewed
     data`, the data will be saved after applying all these
     changes. You can also save the changes in a text file. This is
     useful if you are unsure of the changes you are making, and do
     not want to make permanent modifications or go through relatively
     slow full save of all track data. You can load the original track
     file later and load the change list, and these changes will be
     applied when you play the video.

- `View->Set old track age limit` will allow you to enter the
     number of past frames from which old tracks will be shown when
     the `Show old tracks` menu item is selected. This helps avoid
     visual clutter, but if too short, you will miss a track id that
     was lost from detection for longer than these many frames. The
     optimal number will depend on the quality of the original
     tracking, but starting with something between 200 and 500 may
     help.

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

- Filtering out bad tracks by size: if your objects of interest
     have a specific size range, you can filter out bad detections by
     setting a size limit via `View->Size limits`.
	 
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
