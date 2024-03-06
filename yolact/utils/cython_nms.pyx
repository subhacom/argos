## Note: Figure out the license details later.
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Updated by Subhasis Ray, 2024
#   Notes on update:
#   Switched to using cython memoryviews, but this posed
#   many issues:
#    1. numpy boradcast cannot be done on memoryviews.
#       So needed to keep an ndarray as well.
#       https://stackoverflow.com/questions/66118199/array-broadcasting-in-cython-memoryview
#       For the same reason they must be passed through
#       np.asarray() before arithmetic
#    2. cdef types must be cnp.TYPE_t, but dtype in numpy
#       array constructors must be np.TYPE
#
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as cnp

cdef inline cnp.float32_t max(cnp.float32_t a, cnp.float32_t b) nogil:
    return a if a >= b else b

cdef inline cnp.float32_t min(cnp.float32_t a, cnp.float32_t b) nogil:
    return a if a <= b else b

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def nms(cnp.float32_t[:, :] dets, cnp.float32_t thresh):
    cdef cnp.float32_t[:] x1 = dets[:, 0]
    cdef cnp.float32_t[:] y1 = dets[:, 1]
    cdef cnp.float32_t[:] x2 = dets[:, 2]
    cdef cnp.float32_t[:] y2 = dets[:, 3]
    cdef cnp.float32_t[:] scores = dets[:, 4]

    cdef cnp.float32_t[:] areas = (np.asarray(x2) - np.asarray(x1) + 1) * (np.asarray(y2) - np.asarray(y1) + 1)
    cdef cnp.int32_t[:] order = scores.argsort()[::-1]

    cdef size_t ndets = dets.shape[0]
    suppressed_np = np.zeros((ndets), dtype=np.int32)
    cdef cnp.int32_t[:] suppressed = suppressed_np

    # nominal indices
    cdef size_t _i, _j
    # sorted indices
    cdef size_t i, j
    # temp variables for box i's (the box currently under consideration)
    cdef cnp.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef cnp.float32_t xx1, yy1, xx2, yy2
    cdef cnp.float32_t w, h
    cdef cnp.float32_t inter, ovr

    with nogil:
      for _i in range(ndets):
          i = order[_i]
          if suppressed[i] == 1:
              continue
          ix1 = x1[i]
          iy1 = y1[i]
          ix2 = x2[i]
          iy2 = y2[i]
          iarea = areas[i]
          for _j in range(_i + 1, ndets):
              j = order[_j]
              if suppressed[j] == 1:
                  continue
              xx1 = max(ix1, x1[j])
              yy1 = max(iy1, y1[j])
              xx2 = min(ix2, x2[j])
              yy2 = min(iy2, y2[j])
              w = max(0.0, xx2 - xx1 + 1)
              h = max(0.0, yy2 - yy1 + 1)
              inter = w * h
              ovr = inter / (iarea + areas[j] - inter)
              if ovr >= thresh:
                  suppressed[j] = 1

    return np.where(suppressed_np == 0)[0]
