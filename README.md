# Argos: tracking multiple objects
Argos is a utility for tracking multiple objects (animals) in a video.

# Installation
## Using Anaconda
1. Install [anaconda](https://www.anaconda.com/) python distribution. You can download the free Individual Edition [here](https://www.anaconda.com/products/individual#Downloads)
2. Create an environment with required packages (enter this commands in Anaconda prompt):
    ```commandline
    conda create -n track -c pytorch python cython scipy numpy pyqt pytorch torchvision opencv pyyaml
    ```
    This will create a virtual Python environment called `track`
3. Activate the environment (enter this commands in Anaconda prompt):
    ```commandline
    conda activate track
    ```
4. Install Pytorch. If you want to use CPU-only version, in the Anaconda prompt
    ```commandline
    conda install pytorch torchvision cpuonly -c pytorch
    ```
   If you want GPU support, see [pytorch website](https://pytorch.org/get-started/locally/) to select the right command. But note that you will need to install the appropriate [NVIDIA driver](https://www.nvidia.com/Download/index.aspx) for it to work.  
4. Install pycocotools
    On Windows:
     1. Install [MS Visual Studio Build Tools](https://go.microsoft.com/fwlink/?LinkId=691126). Select Windows XX SDK for your version of Windows.
     2. Go to `C:\Program Files (x86)\Microsoft Visual C++ Build Tools` and run `vcbuildtools_msbuild.bat`
     3. In the Anaconda command prompt run:
    ```commandline
    pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
    ```
   On Linux/Mac you need to have `make` and `g++` installed, and then in the Anaconda command prompt:
   ```commandline
    pip install pycocotools
    ```    
   
5. In the Anaconda prompt, go to where `argos` is unpacked:
    ```commandline
    cd {your_argos_directory}
    ```
    it should have two directories `argos`, and `yolact`.
6. In the Anaconda prompt, update Python path to include this directory:
   ```commandline
    set PYTHONPATH=.;$PYTHONPATH
   ```
   on Windows and
   ```commandline
    export PYTHONPATH=.:$PYTHONPATH
   ```
   on Linux/Unix with `bash` shell.
7. Run `argos` main script:
    ```commandline
    python argos\amain.py
   ```
   on Windows, and
   ```commandline
    python argos/amain.py
   ```
   on Linux/Unix.