#!/bin/env python
import os
import sys
import subprocess
from datetime import datetime

""" Script to run motion tracking on all .avi files in directory
specified.

Usage: python motiontrack_dir.py sourcedir targetdir

If targetdir is not specified, the output files will be produced in
the same directory as source.

Example: 

cd "C:\Users\rays3\Documents\Python Scripts\argos"
set PYTHONPATH=.;%PYTHONPATH
# This script runs this command: python -m argos.capture -m --format=MJPG --threshold=20 -a 10 --interactive=0 --roi=0
# that is, change of 10 contiguous pixels will be considered as movement
python .\motiontrack_dir.py D:\locust_videos_2020\2020_10_25 C:\Users\rays3\Documents\locust_behavior_2020\2020_10_25


Before starting this script you must put the path to folder containing
argos in the PYTHONPATH environment variable.  """


if __name__ == '__main__':
   vid_dir = sys.argv[1]
   out_dir = None
   if len(sys.argv) > 2:
      out_dir = sys.argv[2]
   cmd = 'python -m argos.capture -m --format=MJPG --threshold=20 -a 10 --interactive=0 --roi=0'
   cmd = cmd.split()
   num_processed = 0
   start = datetime.now()
   for dirpath, dirnames, fnames in os.walk(vid_dir):
       print('Walking', dirpath)
       for fname in fnames:
           pre, _, suf = fname.rpartition('.')
           if len(pre) == 0 or suf != 'avi':
              continue
           if out_dir is not None:
              mtracked = os.path.join(out_dir, f'{pre}.mt.avi')
           else:
              mtracked = os.path.join(dirpath, f'{pre}.mt.avi')
           if os.path.exists(mtracked):
              print('Skipping existing target', mtracked)
              continue
           subprocess.check_call(cmd + ['-i', os.path.join(dirpath, fname),
                                                   '-o', mtracked])
           num_processed += 1
   end = datetime.now()
   dt = end - start
   print(f'Processed {num_processed} files in {dt} time.')
               
       
