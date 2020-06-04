REM Run this script in Anaconda cmd prompt AFTER installing the following:
REM
REM Microsoft VC++ build tools from here (in the installation wizard select
REM Windows SDK for your system):
REM     https://go.microsoft.com/fwlink/?LinkId=691126
REM git from here: https://git-scm.com/downloads

conda create -n track -c pytorch -c conda-forge python cython scipy numpy pyqt opencv pyyaml pytorch torchvision cpuonly
conda activate track
"C:\Program Files (x86)\Microsoft Visual C++ Build Tools\vcbuildtools.bat"
pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
