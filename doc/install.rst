Installation using Anaconda
===========================

1. Install `anaconda <https://www.anaconda.com/>`_ python
   distribution. You can download the free Individual Edition `here
   <https://www.anaconda.com/products/individual#Downloads>`_.
   
2. Create an environment with required packages (enter this commands
   in Anaconda prompt)::

     conda create -n track -c conda-forge python cython scipy numpy scikit-learn pyqt pyyaml matplotlib pandas pytables ffmpeg sortedcontainers
   
   This will create a virtual Python environment called ``track``.
   
3. Activate the environment (enter this commands in Anaconda prompt)::

     conda activate track
   
4. Install OpenCV with contributed modules (required for some recent tracking 
   algorithms, but not part of the main OpenCV distribution available in conda)::

     pip install opencv-contrib-python
   
5. Install PyTorch.

   If you have a CUDA capable GPU, see `pytorch website
   <https://pytorch.org/get-started/locally/>`_ to select the right
   command. But note that you will need to install the appropriate
   `NVIDIA driver <https://www.nvidia.com/Download/index.aspx>`_ for
   it to work.

   In case you do not have a CUDA capable GPU, you have to use
   *CPU-only* version (which can be ~10 times slower), in the Anaconda
   prompt::

     conda install pytorch torchvision cpuonly -c pytorch

6. Install ``pycocotools``.

   On Windows:
     1. Install `MS Visual Studio Build Tools
        <https://go.microsoft.com/fwlink/?LinkId=691126>`_.  Select
        Windows XX SDK for your version of Windows.
     2. Go to ``C:\Program Files (x86)\Microsoft Visual C++ Build
        Tools`` on your computer and run ``vcbuildtools_msbuild.bat``
     3. Install [git](https://git-scm.com/downloads)
     4. In the Anaconda command prompt run::

          pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
		
   On Linux/Unix/Mac you need to have ``make`` and ``g++`` installed, and then in 
   the Anaconda command prompt::

     pip install pycocotools


