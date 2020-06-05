# Argos: tracking multiple objects
Argos is a software utility for tracking multiple objects (animals) in a video.

## Getting started

### Using Anaconda
1. Install [anaconda](https://www.anaconda.com/) python distribution. You can download the free Individual Edition [here](https://www.anaconda.com/products/individual#Downloads)
2. Create an environment with required packages (enter this commands in Anaconda prompt):
   ```commandline
   conda create -n track python cython scipy numpy pyqt opencv pyyaml matplotlib pandas
   ```
    This will create a virtual Python environment called `track`
3. Activate the environment (enter this commands in Anaconda prompt):
   ```commandline
   conda activate track
   ```
4. Install PyTorch. If you want to use CPU-only version, in the Anaconda prompt
   ```commandline
   conda install pytorch torchvision cpuonly -c pytorch
   ```
   If you want GPU support, see [pytorch website](https://pytorch.org/get-started/locally/) to select the right command. But note that you will need to install the appropriate [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) for it to work.  
5. Install `pycocotools`

   On Windows:
     1. Install [MS Visual Studio Build Tools](https://go.microsoft.com/fwlink/?LinkId=691126). Select Windows XX SDK for your version of Windows.
     2. Go to `C:\Program Files (x86)\Microsoft Visual C++ Build Tools` and run `vcbuildtools_msbuild.bat`
     3. Install [git](https://git-scm.com/downloads)
     4. In the Anaconda command prompt run:
        ```commandline
        pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
        ```

   On Linux/Unix/Mac you need to have `make` and `g++` installed, and then in the Anaconda command prompt:
   ```commandline
   pip install pycocotools
   ```
   
6. In the Anaconda prompt, go to where `argos` is unpacked:
   ```commandline
   cd {your_argos_directory}
   ```
    it should have these directories there: `argos`, `config`, and `yolact`.
7. In the Anaconda prompt, update Python path to include this directory:

   on Windows command prompt:
   ```commandline
   set PYTHONPATH=.;$PYTHONPATH
   ```
   and on Linux/Unix/Mac with `bash` shell:
   ```commandline
   export PYTHONPATH=.:$PYTHONPATH
   ```
   
8. Run `argos` main script on the Anaconda prompt:

   on Windows: 
   ```commandline
   python argos\amain.py
   ```
   and on Linux/Unix/Mac:
   ```commandline
   python argos/amain.py
   ```
 9. Open the video file using either the `File` menu. After selecting the video file, you will be prompted to:
    1. Select output data directory. 
    2. Select Yolact configuration file, go to the `config` directory inside argos directory and select `yolact.yml`.
    3. File containing trained network weights, and here you should select the `babylocust_resnet101_119999_240000.pth` file.
    
 10. Start tracking: click the `play/pause` button and you should see the tracked locusts. The data will be saved in the directory you chose in step above. 
 
     The bounding boxes of the segmented objects will be saved in `{videofile}.seg.csv` with each row containing `frame-no,x,y,w,h` where (x, y) is the coordinate of the top left corner of the bounding box and `w` and `h` are its width and height respectively. 
     
     The tracks will be saved in `{videofile}.trk.csv`. Each row in this file contains `frame-no,track-id,x,y,w,h`.
     

## Additional utilities
- `capture.py` : a python script to record from a webcam or convert an existing video based on movement. For very long recordings it may be wasteful to record video when there is nothing happening. You can use this script to record or convert video so only parts where there is some minimum change of pixels (for example due to movement) are stored. Alongside the output video, it keeps a `.csv` file with the time of each frame. Check the source code or enetre `python argos/capture.py -h` to find the command line arguments.

- `plot_tracks.py` : a python script with functions to display the tracks.
