import numpy as np
import torch
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[float, ndim=1] boost_calculate_APs_impl(
    np.ndarray[float, ndim=1] out_ious,
    np.ndarray[long, ndim=1] out_tgts,
    np.ndarray[float, ndim=1] mAP_IoUs,
    int num_gt,
):
    cdef int n = out_tgts.shape[0]
    cdef int k = mAP_IoUs.shape[0]
    cdef np.ndarray[double, ndim=1] APs = np.empty(k, dtype=float)

    cdef set tp_set
    cdef np.ndarray[long, ndim=1] tp_cnt
    cdef np.ndarray[double, ndim=1] pr, rc

    for i in range(k):
        tp_set = set()
        tp_cnt = np.empty(n, dtype=long)
        for j in range(n):
            if out_ious[j] > mAP_IoUs[i]:
                tp_set.add(out_tgts[j])
            tp_cnt[j] = len(tp_set)

        pr = tp_cnt / np.arange(1, n + 1, dtype=float)
        rc = tp_cnt / num_gt
        rc[1:] = rc[1:] - rc[:-1]
        APs[i] = (rc * pr).sum()

    return APs.astype(float)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.uint8_t, ndim=1] boost_suppression_impl(
    np.ndarray[float, ndim=2] ious,
    float threshold,
):
    cdef int n = ious.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=1] suppressed = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        if suppressed[i]:
            continue
        suppressed = suppressed | (ious[i] > threshold)
        suppressed[i] = False

    return suppressed


def boost_calculate_APs(
    out_ious: torch.Tensor,
    out_tgts: torch.Tensor,
    mAP_IoUs: torch.Tensor,
    num_gt: int,
):
    return torch.from_numpy(boost_calculate_APs_impl(
        out_ious.float().numpy(),
        out_tgts.long().numpy(),
        mAP_IoUs.float().numpy(),
        num_gt,
    )).float().to(out_ious.device)


def boost_suppression(
    ious: torch.Tensor,
    threshold: float,
):
    return torch.from_numpy(boost_suppression_impl(
        ious.float().cpu().numpy(),
        threshold,
    )).bool().to(ious.device)
