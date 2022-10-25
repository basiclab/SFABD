import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int64_t, ndim=1] calculate_tp_cnt(
    np.ndarray[np.int64_t, ndim=1] tgt_idxs,
    np.ndarray[np.float32_t, ndim=1] out_ious,
    float iou_threshold,
):
    cdef int n = tgt_idxs.shape[0]
    cdef np.ndarray[np.int64_t, ndim=1] tp_cnt = np.zeros(n, dtype=np.int64)
    cdef set tp_set = set()
    for i in range(n):
        if out_ious[i] > iou_threshold:
            tp_set.add(tgt_idxs[i])
        tp_cnt[i] = len(tp_set)
    return tp_cnt
