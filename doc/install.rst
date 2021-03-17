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
		
          pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

	
   On Linux/Unix/Mac you need to have ``make`` and ``g++`` installed, and then in 
   the Anaconda command prompt::

     pip install pycocotools


7. Finally, install the argos toolkit including the tracker with these commands::

       pip install argos-toolkit
       pip install argos-tracker

8. Download pretrained models for testing and for training.
   
   To try Argos tracking on objects in COCO dataset, download the
   pretrained model released with YOLACT
   `here <https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing>`_
   or go to `YOLACT repository <https://github.com/dbolya/yolact>`_ to
   find a mirror.  The corresponding configuration file is already
   installed in
   ``{your_python_environment}/lib/site-packages/argos/config/yolact_base/yolact_base_config.yml``.
   If you used Anaconda as described here,
   ``{your_python_environment}`` should be
   ``C:\Users\{username}\Anaconda3\env\track\`` for Anaconda3 on
   Windows, ``~/.conda/envs/track`` on Linux.

   To train on your own images, use this backbone distributed with
   YOLACT:
   `resnet101_reducedfc.pth <https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing>`_. Argos
   Annotation tool will generate the corresponding configuration for
   you.
   
