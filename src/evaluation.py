from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm

from src.utils import iou, nms


def metric_name(rec_n: int, iou_v: float) -> str:
    return f'R@{rec_n:d},IoU={iou_v:.01f}'


def evaluate(scores2ds, moments, idxs, rec_metrics, nms_threshold):
    max_n = torch.tensor(rec_metrics).max().item()
    output_moments = nms(scores2ds, nms_threshold, pad=max_n)   # [S, max_n, 2]
    ious = iou(moments, output_moments)                         # [S, max_n]
    max_ious = torch.stack([
        ious[:, :rec_n].max(dim=1)[0]
        for rec_n in rec_metrics
    ], dim=1)                                                   # [S, R]
    return {
        'idx': idxs,
        'max_ious': max_ious,
        'best_moment': output_moments[:, 0],
    }


def calculate_recall(
    results: List[Dict[str, torch.Tensor]],
    rec_metrics: List[float],
    iou_metrics: List[float],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    """
    Returns:
        recalls:
        {
            "R@1,IoU0.5": 0.64,
            "R@1,IoU0.7": 0.47,
            "R@5,IoU0.5": 0.56,
            "R@5,IoU0.7": 0.83,
            ...
        }
    """
    rec_metrics = torch.tensor(rec_metrics)
    iou_metrics = torch.tensor(iou_metrics)
    recall_table = torch.zeros(rec_metrics.shape[0], iou_metrics.shape[0])
    num_instance = 0

    for batch in tqdm(results, ncols=0, leave=False, desc="Evaluating"):
        for max_ious in batch['max_ious']:
            num_instance += 1
            # R@{rec_n},IoU={iou_v}
            for rec_idx, rec_n in enumerate(rec_metrics):
                for iou_idx, iou_v in enumerate(iou_metrics):
                    if max_ious[rec_idx] >= iou_v:
                        recall_table[rec_idx, iou_idx] += 1

    recall_table = recall_table / num_instance
    recall = {}
    for i, rec_n in enumerate(rec_metrics):
        for j, iou_v in enumerate(iou_metrics):
            recall[metric_name(rec_n, iou_v)] = recall_table[i, j].item()

    return recall
