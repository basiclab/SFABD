from typing import List, Union, Tuple

import torch
import torch.nn.functional as F


def iou(
    moments1: torch.Tensor,             # [B, 2]
    moments2: torch.Tensor              # [B, n, 2] or [1, n, 2]
) -> torch.Tensor:                      # [B, n]
    """Batch IoU calculation."""
    st1 = moments1[:, 0:1]              # [B, 1]
    ed1 = moments1[:, 1:2]              # [B, 1]
    st2 = moments2[..., 0]              # [B, n]
    ed2 = moments2[..., 1]              # [B, n]
    inter = torch.minimum(ed1, ed2) - torch.maximum(st1, st2)
    union = torch.maximum(ed1, ed2) - torch.minimum(st1, st2)
    return inter.clamp(min=0) / union   # [B, n]


def nms(
    scores2ds: torch.Tensor,                        # [B, N, N]
    mask2d: torch.Tensor,                           # [N, N]
    threshold: float,
    pad: int = None,
) -> Union[List[torch.Tensor], torch.Tensor]:       # List of [?, 2], [B, pad, 2]
    """Batch non-maximum suppression.

    Returns:
        Batch of list of moments that are not suppressed if `pad` is None.
        Otherwise, return a tensor of shape [B, pad, 2].
    """
    B, N, _ = scores2ds.shape
    P = (N + 1) * N // 2

    # shared suppresseion pattern
    moments = mask2d.nonzero()                                              # [P, 2]
    scores1ds = scores2ds[:, moments[:, 0], moments[:, 1]]                  # [B, P]
    moments[:, 1] += 1                                                      # [P, 2]
    moments = moments / N                                                   # [P, 2]
    ious = iou(moments, moments.unsqueeze(0))                               # [P, P]
    ious_mask = ious > threshold
    ious_mask[range(P), range(P)] = False                                   # [P, P]

    # nms
    ranks = scores1ds.argsort(dim=1, descending=True)
    suppressed = scores1ds.new_zeros(B, P, dtype=torch.bool)                # [B, P]
    # iterate from highest score to lowest score
    for ith in range(P):
        moment_idx = ranks[:, ith]                                          # [B]
        moment_suppressed = suppressed[torch.arange(B), moment_idx]         # [B]
        moment_not_suppressed = ~moment_suppressed.unsqueeze(1)             # [B, 1]
        next_suppressed = ious_mask[moment_idx] & moment_not_suppressed     # [B, P]
        suppressed = suppressed | next_suppressed

    nms_moments = []
    nms_scores1ds = []
    for i in range(B):
        rank = ranks[i]
        remain_mask = ~suppressed[i]
        remain_rank = rank[remain_mask[rank]]
        nms_moments.append(moments[remain_rank])
        nms_scores1ds.append(scores1ds[i, remain_rank])

    # padding or truncate
    if pad is not None:
        for i in range(B):
            nms_moments[i] = nms_moments[i][:pad]
            nms_moments[i] = torch.nn.functional.pad(
                nms_moments[i], (0, 0, 0, pad - nms_moments[i].shape[0]))
            nms_scores1ds[i] = nms_scores1ds[i][:pad]
            nms_scores1ds[i] = torch.nn.functional.pad(
                nms_scores1ds[i], (0, pad - nms_scores1ds[i].shape[0]))
        nms_moments = torch.stack(nms_moments, dim=0)
        nms_scores1ds = torch.stack(nms_scores1ds, dim=0)

    return nms_moments, nms_scores1ds


def scores2ds_to_scores1ds(
    scores2ds: torch.Tensor,                # [B, N, N]
    mask2d: torch.Tensor,                   # [N, N]
) -> Tuple[torch.Tensor, torch.Tensor]:     # [P, 2]
    _, N, _ = scores2ds.shape
    moments = mask2d.nonzero()                                          # [P, 2]
    scores1ds = scores2ds[:, moments[:, 0], moments[:, 1]]              # [B, P]
    moments[:, 1] += 1                                                  # [P, 2]
    moments = moments / N                                               # [P, 2]
    return scores1ds, moments


def moments_to_iou2ds(
    target_moment: torch.Tensor,    # [B, 2]
    num_clips: int,                 # N = num_clips
) -> torch.Tensor:                  # [B, N, N]
    """ Convert batch moment to iou2d."""
    B, _ = target_moment.shape
    moments = target_moment.new_ones(num_clips, num_clips).nonzero()    # [P, 2]
    moments[:, 1] += 1                                                  # [P, 2]
    moments = moments / num_clips                                       # [P, 2]
    iou2d = iou(target_moment, moments.unsqueeze(0))                    # [B, P]
    iou2d = iou2d.view(B, num_clips, num_clips)                         # [B, N, N]
    assert (iou2d >= 0).all() and (iou2d <= 1).all()
    return iou2d


def iou2ds_to_iou2d(
    iou2ds: torch.Tensor,       # [M. N, N]
    num_targets: torch.Tensor,  # [S]
) -> torch.Tensor:              # [S, N, N]
    """ Aggregrate iou2ds to iou2d."""
    M, N, _ = iou2ds.shape
    S = num_targets.shape[0]
    assert M >= S
    assert M == num_targets.sum().item()

    # UserWarning: scatter_reduce() is in beta and the API may change at any time.
    # scatter_idx = torch.arange(S).to(iou2ds.device)
    # scatter_idx = scatter_idx.repeat_interleave(num_targets)                # [S]
    # scatter_idx = scatter_idx.unsqueeze(1).unsqueeze(1).expand(-1, N, N)    # [S, N, N]
    # iou2d = iou2ds.new_zeros(S, N, N)
    # iou2d.scatter_reduce_(
    #     0, scatter_idx, iou2ds, reduce="amax", include_self=False)

    iou2d = []
    start = 0
    for i in range(S):
        end = start + num_targets[i]
        iou2d.append(iou2ds[start:end].max(dim=0)[0])
        start = end
    return torch.stack(iou2d, dim=0)


def aggregate_feats(
    feats: torch.Tensor,    # [num_src_clips, C]
    num_tgt_clips: int,     # number of target clip
    op_type: str = 'avg',   # 'avg' or 'max'
) -> torch.Tensor:          # [C, num_tgt_clips]
    """Aggregate the feature of video into fixed shape."""
    assert op_type in ['avg', 'max']

    num_src_clips, _ = feats.shape
    idxs = torch.arange(0, num_tgt_clips + 1) / num_tgt_clips * num_src_clips
    idxs = idxs.round().long()
    feats = F.normalize(feats, dim=1)
    feats_bucket = []
    for i in range(num_tgt_clips):
        s, e = idxs[i], idxs[i + 1]
        # to prevent an empty selection, check the indices are valid.
        if s < e:
            if op_type == 'avg':
                feats_bucket.append(feats[s:e].mean(dim=0))
            if op_type == 'max':
                feats_bucket.append(feats[s:e].max(dim=0)[0])
        else:
            feats_bucket.append(feats[s])
    return torch.stack(feats_bucket, dim=1)                     # channel first


if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)

    num_clips = 10

    # test moments_to_iou2ds
    for _ in range(10):
        num_targets = torch.tensor([1, 2, 3, 4])
        moments = torch.rand(num_targets.sum(), 2).sort(dim=1).values
        iou2ds = moments_to_iou2ds(moments, num_clips)
        iou2d = iou2ds_to_iou2d(iou2ds, num_targets)

        assert torch.allclose(iou2d[0], iou2ds[0])
        assert torch.allclose(iou2d[1], iou2ds[1:3].max(dim=0).values)
        assert torch.allclose(iou2d[2], iou2ds[3:6].max(dim=0).values)
        assert torch.allclose(iou2d[3], iou2ds[6:10].max(dim=0).values)

    # test nms
    for _ in range(10):
        scores2ds = torch.rand(16, num_clips, num_clips)
        new_moments = nms(scores2ds, threshold=0.3, pad=5)
