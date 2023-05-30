from typing import List, Dict

import torch
from tqdm import tqdm

import src.dist as dist
from src.utils import iou


def recall_name(rec_n: int, iou_v: float) -> str:
    return f'R@{rec_n:d},IoU={iou_v:.1f}'


def multi_recall_name(rec_n: int, iou_v: float) -> str:
    return f'R@({rec_n:d}/5),IoU={iou_v:.1f}'


def mAP_name(iou_threshold: float) -> str:
    return f'mAP@{iou_threshold:.2f}'


def batchs2results(
    batchs: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Concatenate list of batch results."""
    out = {}
    for key in batchs[0].keys():
        out[key] = torch.cat([d[key] for d in batchs], dim=0)
    return out


def calculate_recall_worker(
    tgt_moments: torch.Tensor,              # [M, 2]
    out_moments: torch.Tensor,              # [N, 2]
    pad: int,
) -> torch.Tensor:                          # [M, pad]
    ious = iou(
        tgt_moments,                        # [M, 2]
        out_moments[:pad],                  # [N, 2]
    )                                       # [M, ?]
    ious = torch.nn.functional.pad(ious, (0, pad - ious.shape[1]))
    return ious                             # [M, pad]


def calculate_recall(
    pred_moments: List[Dict[str, torch.Tensor]],
    true_moments: List[Dict[str, torch.Tensor]],
    recall_Ns: List[float],
    recall_IoUs: List[float],
) -> Dict[str, float]:
    """
    Returns: {
        "R@1,IoU0.5": 0.64,
        "R@1,IoU0.7": 0.47,
        "R@5,IoU0.5": 0.56,
        "R@5,IoU0.7": 0.83,
        ...
    }
    """
    pred_moments = batchs2results(pred_moments)
    out_moments = pred_moments['out_moments']
    num_proposals = pred_moments['num_proposals']

    true_moments = batchs2results(true_moments)
    tgt_moments = true_moments['tgt_moments']
    num_targets = true_moments['num_targets']

    calc_buffer = []
    shift_p = 0
    shift_t = 0
    for num_p, num_t in zip(num_proposals, num_targets):
        calc_buffer.append([
            tgt_moments[shift_t: shift_t + num_t],
            out_moments[shift_p: shift_p + num_p],
        ])
        shift_p += num_p
        shift_t += num_t

    max_N = max(recall_Ns)
    ious = []
    for data in tqdm(calc_buffer, ncols=0, leave=False, desc="Recall",
                     disable=not dist.is_main()):
        ious.append(calculate_recall_worker(*data, pad=max_N))
    ious = torch.cat(ious)

    recall = {}
    for recall_n in recall_Ns:
        for recall_iou in recall_IoUs:
            max_ious = ious[:, :recall_n].max(dim=1).values
            tp = (max_ious >= recall_iou).long().sum().item()
            recall[recall_name(recall_n, recall_iou)] = tp / len(ious)

    return recall


def calculate_multi_recall(
    pred_moments: List[Dict[str, torch.Tensor]],
    true_moments: List[Dict[str, torch.Tensor]],
    recall_Ns: List[float],
    recall_IoUs: List[float],
) -> Dict[str, float]:
    """
    Returns: {
        "R@(5,5),IoU0.5": 0.xx,
        "R@(5,5),IoU0.7": 0.xx,
        ...
    }
    """
    pred_moments = batchs2results(pred_moments)
    out_moments = pred_moments['out_moments']
    num_proposals = pred_moments['num_proposals']

    true_moments = batchs2results(true_moments)
    tgt_moments = true_moments['tgt_moments']
    num_targets = true_moments['num_targets']

    calc_buffer = []
    shift_p = 0
    shift_t = 0
    for num_p, num_t in zip(num_proposals, num_targets):
        calc_buffer.append([
            tgt_moments[shift_t: shift_t + num_t],
            out_moments[shift_p: shift_p + num_p],
        ])
        shift_p += num_p
        shift_t += num_t

    max_N = max(recall_Ns)
    ious = []
    for data in tqdm(calc_buffer, ncols=0, leave=False, desc="Recall",
                     disable=not dist.is_main()):
        ious.append(calculate_recall_worker(*data, pad=max_N))
    ious = torch.cat(ious)  # [5 targets * 1000 samples, 5 proposals]
    recall = {}
    for recall_n in recall_Ns:
        for recall_iou in recall_IoUs:
            shift_t = 0
            recall_list = []
            for num_t in num_targets:
                max_ious = ious[shift_t:shift_t + num_t, :recall_n].max(dim=1).values
                rec = float((max_ious >= recall_iou).long().sum().item() / num_t)
                recall_list.append(rec)
                shift_t += num_t
            recall[multi_recall_name(recall_n, recall_iou)] = sum(recall_list) / len(recall_list)

    return recall


def calculate_APs_worker(
    tgt_moments: torch.Tensor,                      # [M, 2]
    out_moments: torch.Tensor,                      # [N, 2]
    out_scores1ds: torch.Tensor,                    # [N]
    mAP_ious: List[float],                          # [K]
    max_proposals: int,
) -> torch.Tensor:                                  # [K]
    ious = iou(
        out_moments,                                # [N, 2]
        tgt_moments,                                # [M, 2]
    )                                               # [N, M]
    out_ious, out_tgts = ious.max(dim=1)            # [N], [N]
    order = out_scores1ds.argsort(descending=True)  # [N]
    out_ious = out_ious[order][:max_proposals]      # [N]
    out_tgts = out_tgts[order][:max_proposals]      # [N]

    try:
        # 10x faster
        from boost import boost_calculate_APs
        APs = boost_calculate_APs(
            out_ious, out_tgts, mAP_ious, len(tgt_moments))
    except Exception as e:
        APs = []
        for mAP_iou in mAP_ious:
            tp_cnt = []
            tp_set = set()
            for out_tgt, out_iou in zip(out_tgts, out_ious):
                if out_iou >= mAP_iou:
                    tp_set.add(out_tgt.item())
                tp_cnt.append(len(tp_set))
            tp_cnt = torch.tensor(tp_cnt)

            pr = tp_cnt / torch.arange(1, len(tp_cnt) + 1, dtype=torch.float32)
            rc = tp_cnt / len(tgt_moments)

            rc[1:] = rc[1:] - rc[:-1]
            AP = (rc * pr).sum()
            APs.append(AP)
        APs = torch.tensor(APs)

    return APs


def calculate_mAPs(
    pred_moments: List[Dict[str, torch.Tensor]],
    true_moments: List[Dict[str, torch.Tensor]],
    mAP_ious: List[float] = torch.linspace(0.5, 0.95, 10),
    max_proposals: int = 10,
) -> float:
    pred_moments = batchs2results(pred_moments)
    out_moments = pred_moments['out_moments']
    out_scores1ds = pred_moments['out_scores1ds']
    num_proposals = pred_moments['num_proposals']

    true_moments = batchs2results(true_moments)
    tgt_moments = true_moments['tgt_moments']
    num_targets = true_moments['num_targets']

    # buffer for all samples
    calc_buffer = []
    shift_p = 0
    shift_t = 0
    for num_p, num_t in zip(num_proposals, num_targets):
        sample_tgt_moments = tgt_moments[shift_t: shift_t + num_t]
        calc_buffer.append([
            sample_tgt_moments,
            out_moments[shift_p: shift_p + num_p],
            out_scores1ds[shift_p: shift_p + num_p],
        ])

        shift_p += num_p
        shift_t += num_t

    # calculate AP for mixed-length clips
    APs = []
    for data in tqdm(calc_buffer, ncols=0, leave=False, desc="mAP",
                     disable=not dist.is_main()):
        APs.append(calculate_APs_worker(*data, mAP_ious, max_proposals))
    APs = torch.stack(APs)

    APs_dict = {
        'all': APs,                       # all
        'sgl': APs[num_targets == 1],     # single-target
        'mul': APs[num_targets > 1],      # multi-target
    }

    mAPs_dict = {}
    for name, APs in APs_dict.items():
        if len(APs) != 0:
            mAPs = APs.mean(dim=0)
            for mAP, iou_threshold in zip(mAPs, mAP_ious):
                mAPs_dict[f"{name}/{mAP_name(iou_threshold)}"] = mAP.item()
            mAPs_dict[f"{name}/mAP@avg"] = APs.mean().item()
        else:
            for iou_threshold in mAP_ious:
                mAPs_dict[f"{name}/{mAP_name(iou_threshold)}"] = 0.0
            mAPs_dict[f"{name}/mAP@avg"] = 0.0

    results = {}
    for name in APs_dict.keys():
        for target_iou in [0.5, 0.75]:
            key = f"{name}/{mAP_name(target_iou)}"
            results[key] = mAPs_dict[key]
        key = f"{name}/mAP@avg"
        results[key] = mAPs_dict[key]

    return results


def calculate_mAPs_no_mean(
    pred_moments: List[Dict[str, torch.Tensor]],
    true_moments: List[Dict[str, torch.Tensor]],
    mAP_ious: List[float] = torch.linspace(0.5, 0.95, 10),
    max_proposals: int = 10,
) -> float:
    pred_moments = batchs2results(pred_moments)
    out_moments = pred_moments['out_moments']
    out_scores1ds = pred_moments['out_scores1ds']
    num_proposals = pred_moments['num_proposals']

    true_moments = batchs2results(true_moments)
    tgt_moments = true_moments['tgt_moments']
    num_targets = true_moments['num_targets']

    # buffer for all samples
    calc_buffer = []
    shift_p = 0
    shift_t = 0
    for num_p, num_t in zip(num_proposals, num_targets):
        sample_tgt_moments = tgt_moments[shift_t: shift_t + num_t]
        calc_buffer.append([
            sample_tgt_moments,
            out_moments[shift_p: shift_p + num_p],
            out_scores1ds[shift_p: shift_p + num_p],
        ])

        shift_p += num_p
        shift_t += num_t

    # calculate AP for mixed-length clips
    APs = []
    for data in tqdm(calc_buffer, ncols=0, leave=False, desc="mAP",
                     disable=not dist.is_main()):
        APs.append(calculate_APs_worker(*data, mAP_ious, max_proposals))
    APs = torch.stack(APs)   # [num_samples, num_iou_thres]  ex. [1550, 10]

    # return every samples' mAP_avg
    return APs.mean(dim=1), num_targets  # [1550]
