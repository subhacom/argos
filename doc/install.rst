Installation using Anaconda
===========================

1. Install `anaconda <https://www.anaconda.com/>`_ python
   distribution. You can download the free Individual Edition here:
   https://www.anaconda.com/products/individual#Downloads.
   
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

   *NOTE If you face issues with conda claiming package dependencies are unresolvable, a workaround can be installing pytorch as part of creating the new environment in step 2*::

     conda create -n track -c conda-forge -c pytorch -c nvidia python cython scipy numpy scikit-learn pyqt pyyaml matplotlib pandas pytables ffmpeg sortedcontainers pytorch torchvision torchaudio pytorch-cuda=12.1
     
6. Install ``pycocotools``.

   On Windows:
     
   1. Install MS Visual Studio Build Tools. The installer for Visual
      Studio 2019 Build Tools is available here:
      https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019

      You can skip this if you have a functioning Visual C++
      installation with the build tools >= 14.0.
      
   2. Install git from here: https://git-scm.com/downloads or enter in the Anaconda command prompt::

	conda install git
	
   3. In the Anaconda command prompt run (after `conda activate track`)::

	  "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

      or the appropriate .bat file for your installation (see
      https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-160#use-the-developer-tools-in-an-existing-command-window). The
      exact location of this file can vary between installations. You
      can run it using the following steps (for 64 bit systems):
	
      1. Go to `Start menu-> Visual Studio 2019`
      2. Right-click `x64 Native Tools Command Prompt` and in the popup menu select `More->Open File Location`.
      3. In the folder that opens, right click on the `x64 Native ...` shortcut and select `Properties`.
      4. Copy the `Target` field and paste it in the Anaconda command
         prompt and press `Enter`.

   4. In the same prompt run::
	  
          pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

      If this throws this error::

	fatal: unable to access 'https://github.com/philferriere/cocoapi.git/': SSL certificate problem: unable to get local issuer certificate

      then you may be able to resolve this by entering the following in the Anaconda prompt::

	git config --global http.sslbackend schannel

      and try::

          pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

      again.

   On Linux/Unix/Mac you need to have `make` and `g++` installed, and
   then in the Anaconda command prompt enter::

     pip install pycocotools

.. seealso::

      - https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/62381
      - https://docs.microsoft.com/en-us/answers/questions/136595/error-microsoft-visual-c-140-or-greater-is-require.html
      - https://stackoverflow.com/questions/23885449/unable-to-resolve-unable-to-get-local-issuer-certificate-using-git-on-windows

7. Finally, install the argos toolkit and the tracker with these commands::

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
   

Installation using venv (useful on Mac)
=======================================

On Mac: you can use venv module to create virtual environment like conda (this does not require admin access)::

     python3 -m venv track

     source track/bin/activate

     pip install torch torchvision torchaudio opencv-contrib-python Cython


     
Followed by::
  
     pip install pycocotools argos-toolkit argos-tracker

   
If you have Mac with Intel CPU, and encounter an error after the command above, (like `#error: architecture not supported, error: command 'clang' failed with exit status 1`) try the following::

     export ARCHFLAGS="-arch x86_64"

     CC=clang CXX=clang++ python -m pip install pycocotools argos-toolkit argos-tracker
   


After this, try running the review tool::

    python -m argos.review


Installing DCNv2 for YOLACT++
=============================

YOLACT++ is an improved version of YOLACT that uses DCNv2 library for
Deformable Convolution Network. This library comes with YOLACT source
code (``yoact/external/DCNv2``). You can install it with pip:

``pip install DCNv2``.

To build this library on your own you need CUDA toolkit from NVidia
installed. Also, on MS Windows you need the Visual Studio Build Tools
described above. After that

- First activate your conda environment where YOLACT and Argos are
  installed.

- Change directory to ``yoact/external/DCNv2``.
  
- Run ``python setup.py build develop``

- Run ``pip install .``


You can find some details in the YOLACT README file.
