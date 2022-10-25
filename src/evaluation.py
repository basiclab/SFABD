from typing import List, Dict, Tuple

import torch

from src.utils import iou, nms


def recall_name(rec_n: int, iou_v: float) -> str:
    return f'R@{rec_n:d},IoU={iou_v:.1f}'


def mAP_name(tp_threshold: float) -> str:
    return f'mAP@{tp_threshold:.2f}'


def batchs2results(
    batchs: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Concatenate list of batch results."""
    out = {}
    for key in batchs[0].keys():
        out[key] = torch.cat([d[key] for d in batchs], dim=0)
    return out


def prepare_recall(
    scores2ds: torch.Tensor,    # [S, N, N]
    tgt_moments: torch.Tensor,  # [M, 2]
    num_targets: torch.Tensor,  # [S]
    mask2d: torch.Tensor,
    nms_threshold: float,
    num_moments: List[float],
) -> Dict[str, torch.Tensor]:
    max_n = torch.tensor(num_moments).max().item()
    out_moments, _ = nms(scores2ds, mask2d, nms_threshold, pad=max_n)   # [S, max_n, 2]
    out_moments = out_moments.repeat_interleave(num_targets, dim=0)     # [M, max_n, 2]
    ious = iou(
        tgt_moments,                                                    # [M, 2]
        out_moments,                                                    # [M, max_n, 2]
    )                                                                   # [M, max_n]
    return {
        'ious': ious,
        'out_moments': out_moments,
    }


def prepare_mAP(
    scores2ds: torch.Tensor,    # [S, N, N]
    tgt_moments: torch.Tensor,  # [M, 2]
    num_targets: torch.Tensor,  # [S]
    mask2d: torch.Tensor,
    nms_threshold: float,
) -> Dict[str, torch.Tensor]:
    """Calculate max iou and corresponding GT id for each proposal."""
    M = num_targets.sum()
    assert M == tgt_moments.shape[0]

    out_moments, scores1ds = nms(
        scores2ds, mask2d, nms_threshold)                       # List[S, [n, 2]], List[S, [n]]                                                          # [M, P]

    tgt_idxs = []
    out_ious = []
    num_proposals = []
    shift = 0
    for i, num in enumerate(num_targets):
        p, _ = out_moments[i].shape
        ious = iou(
            out_moments[i],                                     # [p, 2]
            tgt_moments[shift: shift + num].unsqueeze(0),       # [num, 2]
        )                                                       # [p, num]
        out_iou, tgt_idx = ious.max(dim=1)                      # [p], [p]
        tgt_idxs.append(tgt_idx)
        out_ious.append(out_iou)
        num_proposals.append(p)
        shift += num
    tgt_idxs = torch.cat(tgt_idxs)                              # [sum(n)]
    out_ious = torch.cat(out_ious)                              # [sum(n)]
    scores1ds = torch.cat(scores1ds)                            # [sum(n)]
    num_proposals = torch.tensor(num_proposals)                 # [S]
    return {
        'out_conf': scores1ds,
        'tgt_idxs': tgt_idxs,
        'out_ious': out_ious,
        'num_targets': num_targets,
        'num_proposals': num_proposals,
    }


def calculate_recall(
    batchs: List[Dict[str, torch.Tensor]],
    num_moments: List[float],
    iou_thresholds: List[float],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    """
    Returns: {
        "R@1,IoU0.5": 0.64,
        "R@1,IoU0.7": 0.47,
        "R@5,IoU0.5": 0.56,
        "R@5,IoU0.7": 0.83,
        ...
    }
    """
    results = batchs2results(batchs)
    ious = results['ious']    # [M, max_n]
    recall = {}
    for rec_n in num_moments:
        for iou_v in iou_thresholds:
            max_ious = ious[:, :rec_n].max(dim=1).values
            tp = (max_ious > iou_v).long().sum().cpu().item()
            recall[recall_name(rec_n, iou_v)] = tp / len(ious)

    return recall


def calculate_mAP(
    batchs: List[Dict[str, torch.Tensor]],
    tp_thresholds: List[float] = torch.linspace(0.5, 0.95, 10).tolist(),
) -> float:
    results = batchs2results(batchs)
    AP_sum = 0
    mAPs = {}
    for tp_threshold in tp_thresholds:
        mAP = calculate_AP(results, tp_threshold)
        mAPs[mAP_name(tp_threshold)] = mAP
        AP_sum += mAP
    avg_mAP = AP_sum / len(tp_thresholds)
    return {
        mAP_name(0.5): mAPs[mAP_name(0.5)],
        mAP_name(0.75): mAPs[mAP_name(0.75)],
        'avg mAP': avg_mAP,
    }


def calculate_AP(
    results: Dict[str, torch.Tensor],
    tp_threshold: float,
) -> float:
    out_conf = results['out_conf']              # [P]
    tgt_idxs = results['tgt_idxs']              # [P]
    out_ious = results['out_ious']              # [P]
    num_targets = results['num_targets']        # [S]
    num_proposals = results['num_proposals']    # [S]
    assert out_conf.shape == tgt_idxs.shape == out_ious.shape
    assert num_targets.shape == num_proposals.shape
    assert out_conf.shape[0] == num_proposals.sum()

    shift_t = 0
    shift_p = 0
    for num_t, num_p in zip(num_targets, num_proposals):
        tgt_idxs[shift_p: shift_p + num_p] += shift_t
        shift_t = shift_t + num_t
        shift_p = shift_p + num_p

    order = out_conf.argsort(descending=True)
    tgt_idxs = tgt_idxs[order]
    out_ious = out_ious[order]
    tp_cnt = get_tp_cnt(tgt_idxs, out_ious, tp_threshold)

    pr = tp_cnt / torch.arange(1, len(tp_cnt) + 1, dtype=torch.float32)
    rc = tp_cnt / num_targets.sum()

    # riemann sum
    rc[1:] = rc[1:] - rc[:-1]
    auc = (rc * pr).sum()

    return auc.cpu().item()


def calculate_tp_cnt(
    tgt_idxs: torch.Tensor,     # [P]
    out_ious: torch.Tensor,     # [P]
    tp_threshold: float
):
    tp_cnt = []
    tp_set = set()
    for idx, iou in zip(tgt_idxs, out_ious):
        if iou > tp_threshold:
            tp_set.add(idx.cpu().item())
        tp_cnt.append(len(tp_set))
    tp_cnt = torch.tensor(tp_cnt)
    return tp_cnt


def get_tp_cnt(
    tgt_idxs: torch.Tensor,     # [P]
    out_ious: torch.Tensor,     # [P]
    tp_threshold: float,
) -> torch.Tensor:              # [P]
    try:
        # 1500 times faster
        from boost import calculate_tp_cnt as boost_calculate_tp_cnt
        tp_cnt = torch.from_numpy(boost_calculate_tp_cnt(
            tgt_idxs.cpu().numpy(), out_ious.cpu().numpy(), tp_threshold))
    except Exception:
        # python implementation
        tp_cnt = calculate_tp_cnt(tgt_idxs, out_ious, tp_threshold)
    return tp_cnt
