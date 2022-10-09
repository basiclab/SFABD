from typing import List, Union

import torch


def iou(
    target: torch.Tensor,           # [B, 2]
    moments: torch.Tensor           # [B, N, 2]
) -> torch.Tensor:
    """Batch IoU calculation.

    Returns:
        iou: shape = [B, N]
    """
    st = target[:, 0:1]                 # [B, 1]
    ed = target[:, 1:2]                 # [B, 1]
    moments_st = moments[..., 0]        # [B, ?]
    moments_ed = moments[..., 1]        # [B, ?]
    inter = torch.minimum(moments_ed, ed) - torch.maximum(moments_st, st)
    union = torch.maximum(moments_ed, ed) - torch.minimum(moments_st, st)
    return inter.clamp(min=0) / union   # [B, ?]


def nms(
    scores2d: torch.Tensor,     # [B, N, N]
    threshold: float,
    pad: int = None,
) -> Union[List[torch.Tensor], torch.Tensor]:        # List of [?, 2]
    """Batch non-maximum suppression.

    Returns:
        Batch of list of moments that are not suppressed if `pad` is None.
        Otherwise, return a tensor of shape [B, pad, 2].
    """
    B, N, _ = scores2d.shape
    P = (N + 1) * N // 2

    # shared suppresseion pattern
    moments = scores2d.new_ones(N, N).triu().nonzero()              # [P, 2]
    scores1d = scores2d[:, moments[:, 0], moments[:, 1]]            # [B, P]
    moments[:, 1] += 1                                              # [P, 2]
    moments = moments / N                                           # [P, 2]
    ious = iou(moments, moments.unsqueeze(0).expand(P, -1, -1))     # [P, P]
    ious_mask = ious > threshold
    ious_mask[range(P), range(P)] = False                           # [P, P]

    # nms
    ranks = scores1d.argsort(dim=1, descending=True)
    suppressed = scores1d.new_zeros(B, P, dtype=torch.bool)                 # [B, P]
    for ith in range(P):        # iterate from highest score to lowest score
        moment_idx = ranks[:, ith]                                          # [B]
        moment_suppressed = suppressed[torch.arange(B), moment_idx]         # [B]
        moment_not_suppressed = ~moment_suppressed.unsqueeze(1)             # [B, 1]
        next_suppressed = ious_mask[moment_idx] & moment_not_suppressed     # [B, P]
        suppressed = suppressed | next_suppressed

    remain_moments = []
    for i in range(B):
        rank = ranks[i]
        remain_mask = ~suppressed[i]
        remain_moments.append(moments[rank[remain_mask[rank]]])

    if pad is not None:
        for i in range(B):
            remain_moments[i] = remain_moments[i][:pad]
            remain_moments[i] = torch.nn.functional.pad(
                remain_moments[i], (0, 0, 0, pad - remain_moments[i].shape[0]))
        remain_moments = torch.stack(remain_moments, dim=0)

    return remain_moments


def moment_to_iou2d(
    target_moment: torch.Tensor,    # [B, 2]
    num_clips: int,
) -> torch.Tensor:
    """ Convert batch moment to iou2d.

    Returns:
        iou2d: [B, D, D]
    """
    B, _ = target_moment.shape
    moments = target_moment.new_ones(num_clips, num_clips).nonzero()    # [P, 2]
    moments[:, 1] += 1                                                  # [P, 2]
    moments = moments / num_clips                                       # [P, 2]
    moments = moments.unsqueeze(0).expand(B, -1, -1)                    # [B, P, 2]
    iou2d = iou(target_moment, moments).reshape(B, num_clips, num_clips)
    assert (iou2d >= 0).all() and (iou2d <= 1).all()
    return iou2d


def piror(dataset: torch.utils.data.Dataset):
    """Calculate the prior distribution of the dataset.

    Returns:
        prior: [num_clips, num_clips]
    """
    iou2ds_all = []
    for data_dict in dataset:
        iou2ds_all.append(data_dict['iou2ds'])
    iou2ds = torch.cat(iou2ds_all, dim=0)                       # [?, D, D]
    N, D, _ = iou2ds.shape
    iou2ds = iou2ds.view(N, -1)
    mask = iou2ds == iou2ds.max(dim=1, keepdim=True).values     # [N, D*D]
    heatmap = mask.float().view(D, D).sum(dim=0)                # [D, D]
    heatmap = heatmap / heatmap.max()                           # [D, D]

    assert (mask.sum(dim=1) == 1).all()  # each iou2d has only one max value
    return heatmap


if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)

    num_clips = 10
    duration = 29.7

    # test moment_to_iou2d
    for _ in range(10):
        target_moment = torch.rand(16, 2) * duration
        target_moment = target_moment.sort(dim=1).values
        iou2ds = moment_to_iou2d(target_moment / duration, num_clips)

    # test nms
    for _ in range(10):
        scores2d = torch.rand(16, num_clips, num_clips)
        new_moments = nms(scores2d, threshold=0.3, pad=5)
