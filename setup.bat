:: This does not work as a script. Copy-paste the following commands, one line
:: at a time, in your Anaconda cmd prompt (after cd-ing to this directory)
:: only once to set up everything AFTER installing the following:
::
:: Microsoft VC++ build tools from here (in the installation wizard select
:: Windows SDK for your system):
::     https://go.microsoft.com/fwlink/?LinkId=691126
::
:: git from here: https://git-scm.com/downloads
::
:: Afterwards, run argos in Anaconda prompt after switching to this directory
:: (for example, if you unpacked this in C:\Users\me\Documents\argos, then
:: in the Anaconda prompt enter "cd C:\Users\me\Documents\argos")
:: and enter "python .\argos\amain.py"
::
conda create -n track -c pytorch -c conda-forge python cython scipy numpy pyqt opencv pyyaml pytorch torchvision cpuonly
conda activate track
C:\Program Files (x86)\Microsoft Visual C++ Build Tools\vcbuildtools.bat
pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
