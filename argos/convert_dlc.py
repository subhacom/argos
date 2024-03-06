# convert_dlc.py ---
#
# Filename: convert_to_argos.py
# Description:
# Author: Subhasis Ray
# Created: Mon Mar  4 17:22:46 2024 (+0530)
# Last-Updated: Wed Mar  6 22:20:36 2024 (+0530)
#           By: Subhasis Ray
#

# Code:
"""Convert DLC file into argos format.


One can then use the Argos review tool for track correction.

"""

import os
import sys
import pandas as pd


def load_dlc(fpath):
    """Load data from DLC HDF5 file with path `fpath`

    Parameters
    ----------
    fpath: str
        path of the file containing DLC tracked data

    Returns
    -------
    pd.DataFrame: track data
        This flattens the hierarchical columns in DLC data and
        assigns unique integer IDs to each tracked point.
        Information is not lost, as the `idstr` column keeps
        "{scorer}/{individuals}/{bodyparts}". Another difference from
        argos format at the moment is that instead of `confidence` we
        are keeping the `likelihood` column.

    """
    df = pd.read_hdf(fpath)
    # Get the lowest level of columns - assuming some file identifier
    # and the body part are the upper levels. Assuming tracked single
    # body part.
    trackid = 0  # counts the individual and body part to generate unique ids
    dlist = []
    names = df.columns.names
    if 'individuals' in names:  # multiple individuals
        for key, group in df.T.groupby(['scorer', 'individuals', 'bodyparts']):
            data = group.T.dropna()
            data.columns = [
                names[-1] for names in data.columns.to_flat_index()
            ]
            data['trackid'] = trackid
            data[['individual', 'bodypart']] = key[1:]
            dlist.append(data.reset_index(names=['frame']))
            trackid += 1
    else:  # single animal
        for key, group in df.T.groupby(['scorer', 'bodyparts']):
            data = group.T.dropna()
            data.columns = [
                names[-1] for names in data.columns.to_flat_index()
            ]
            data['trackid'] = trackid
            data['individual'] = '1'
            data['bodypart'] = key[1]
            dlist.append(data.reset_index(names=['frame']))
            trackid += 1

    ret = pd.concat(dlist)
    return ret


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} input output [width height]')
        print('Convert DLC file `input` to Argos format file `output`')
        sys.exit(0)
    infile = sys.argv[1]
    outfile = sys.argv[2]
    if len(sys.argv) == 5:
        width = float(sys.argv[3])
        height = float(sys.argv[4])
    else:  # place holder - DLC does not use bbox
        width = 5.0
        height = 5.0

    data = load_dlc(infile)
    data['w'] = width
    data['h'] = height
    data.to_hdf(outfile, key='tracked')
    print(f'Converted {infile} to {outfile}')

#
# conert_dlc.py ends here
