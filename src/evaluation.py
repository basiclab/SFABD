from typing import List, Dict, Tuple

import torch
from tqdm import tqdm

from src.utils import iou, nms, scores2d_to_moments_scores1d


device = torch.device('cuda:0')


def metric_name(rec_n: int, iou_v: float) -> str:
    return f'R@{rec_n:d},IoU={iou_v:.01f}'


def recall_table_to_dict(
    recall_table: torch.Tensor,     # [num_rec_metrics, num_iou_metrics]
    rec_metrics: float,
    iou_metrics: float,
) -> Dict[str, float]:
    """Convert recall matrix to dict.

    Returns:
        recall:
        {
            "R@1,IoU0.5": 0.64,
            "R@1,IoU0.7": 0.47,
            ...
        }
    """
    assert recall_table.shape == (len(rec_metrics), len(iou_metrics))
    recall = {}
    for i, rec_n in enumerate(rec_metrics):
        for j, iou_v in enumerate(iou_metrics):
            recall[metric_name(rec_n, iou_v)] = recall_table[i, j].item()
    return recall


def evaluate(
    results: List[Dict],
    nms_threshold: float,
    rec_metrics: List[float],
    iou_metrics: List[float],
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    """
    Returns:
        recalls:
        {
            "all-target": {
                "R@1,IoU0.5": 0.64,
                "R@1,IoU0.7": 0.47,
                ...
            },
            "multi-target": ...,
            "1-target": ...,
            "2-target": ...,
            "3-target": ...,
        }
        results:
        [
            {
                "idx": 0,
                "vid": "VXJDG",
                ...
            },
            ...
        ]
    """
    # num_rec_metrics, num_iou_metrics = len(rec_metrics), len(iou_metrics)
    rec_metrics = torch.tensor(rec_metrics)           # [1, 5, 10]
    iou_metrics = torch.tensor(iou_metrics)           # [0.5, 0.7]

    recall_tables = torch.zeros(3, rec_metrics.shape[0], iou_metrics.shape[0])
    num_instances = torch.zeros(3)

    # evaluation results for each prediction
    eval_results = []
    pbar = tqdm(results, ncols=0, leave=False, desc="Evaluating")
    for batch in pbar:
        for batch_idx in range(len(batch['scores2d'])):
            scores2d = batch['scores2d'][batch_idx]
            target_moments = batch['moments'][batch_idx]
            duration = batch['duration'][batch_idx]
            num_target = len(target_moments) - 1
            num_instances[num_target] += 1

            eval_result = {
                'query': batch['query'][batch_idx],
                'sents': batch['sents'][batch_idx],
                'vid': batch['vid'][batch_idx],
                'idx': batch['idx'][batch_idx].item(),
                'target_moments': target_moments.tolist(),
            }

            output_moments, scores1d = scores2d_to_moments_scores1d(scores2d, duration)
            rank = nms(output_moments, scores1d, nms_threshold)
            output_moments = output_moments[rank]

            # R@{rec_n}
            for rec_idx, rec_n in enumerate(rec_metrics):
                max_ious = []                       # max iou for each target
                best_moments = []
                for target_moment in target_moments:
                    ious = iou(target_moment, output_moments[:rec_n])
                    best_moments.append(output_moments[torch.argmax(ious)])
                    max_ious.append(ious.max())
                max_ious = torch.tensor(max_ious)   # [num_target]
                eval_result[f'R@{rec_n:d}'] = {
                    'best_moments': torch.stack(best_moments).tolist(),
                    'max_ious': max_ious.mean().item(),
                }

                # R@{rec_n},IoU={iou_v}
                for iou_idx, iou_v in enumerate(iou_metrics):
                    recall_mask = max_ious >= iou_v
                    recall = recall_mask.float().mean()
                    recall_tables[num_target, rec_idx, iou_idx] += recall

                    eval_result[metric_name(rec_n, iou_v)] = {
                        'recall': recall.item(),
                        'true_positive_mask': recall_mask.tolist(),
                    }
            eval_results.append(eval_result)
    pbar.close()

    recall_table_all_target = recall_tables.sum(0) / num_instances.sum()
    recall_table_multi_target = recall_tables[1:].sum(0) / num_instances[1:].sum()
    for num_target in range(len(recall_tables)):
        recall_tables[num_target] /= num_instances[num_target]

    recalls = {
        "all-target": recall_table_to_dict(
            recall_table_all_target, rec_metrics, iou_metrics),
        "multi-target": recall_table_to_dict(
            recall_table_multi_target, rec_metrics, iou_metrics),
    }
    for num_target, recall_table in enumerate(recall_tables):
        recalls[f"{num_target + 1}-target"] = recall_table_to_dict(
            recall_table, rec_metrics, iou_metrics)

    return recalls, eval_results


def inference_loop(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
) -> List[Dict]:
    model.eval()
    results = []    # batch of scores2d and info. for calculating recall
    for batch, info in tqdm(loader, ncols=0, leave=False, desc="Inferencing"):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            *_, scores2d = model(**batch)
        results.append({
            'scores2d': scores2d.cpu(),
            **info,
        })
    return results
