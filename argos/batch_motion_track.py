# Script to process multiple files in a directory
"""Usage:

python -m argos.batch_motiontrack <input_directory> <output_directory> [<nproc>]

Convert all video files with .avi extension in a directory into motion-triggered
version, using MJPG format for output.

Arguments
---------

input_directory: str
    should have the original videos and corresponding csv
    files of frame timestamps.

output_directory: str
    this is where the motion-triggered videos are dumped
    along with csv files containing the frame timestamps

nproc: int (optional)
    number of parallel processes to create. Default is
    1, meaning the files will be processed sequentially.

"""
import sys
import os
import subprocess


if __name__ == '__main__':
    indir = sys.argv[1]
    outdir = sys.argv[2]
    nproc = 1
    if len(sys.argv) > 3:
        nproc = int(sys.argv[3])
    infiles = [fname for fname in os.listdir(indir) if fname.endswith('.avi')]
    chunks = [infiles[ii:ii+nproc] for ii in range(0, len(infiles), nproc)]
    for flist in chunks:
        proclist = []
        for fname in flist:
            try:
                inpath = os.path.join(indir, fname)
                ofname = fname.rpartition('.')[0]
                outpath = os.path.join(outdir, f'{ofname}.mt.avi')
                proc = subprocess.Popen(['python', '-m', 'argos.capture',
                                         '--roi', '0',
                                         '--format', 'MJPG',
                                         '-i', inpath,
                                         '-o', outpath,
                                         '-m', '-a', '10'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
                proclist.append(proc)
            except KeyboardInterrupt:
                sys.exit(0)
        try:
            for proc in proclist:
                proc.wait()
                res = proc.communicate()
                print('Return code', proc.returncode)
                # print('Result\n', res)
                print('STDERR\n', res[1].decode('utf-8')
        except KeyboardInterrupt:
            sys.exit(0)
