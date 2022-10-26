from typing import List, Union, Tuple

import torch
import torch.nn.functional as F


def iou(
    moments1: torch.Tensor,                                     # [N, 2]
    moments2: torch.Tensor                                      # [M, 2]
) -> torch.Tensor:                                              # [N, M]
    st1 = moments1[:, 0:1]                                      # [N, 1]
    ed1 = moments1[:, 1:2]                                      # [N, 1]
    st2 = moments2[:, 0:1].t()                                  # [1, M]
    ed2 = moments2[:, 1:2].t()                                  # [1, M]
    inter = torch.minimum(ed1, ed2) - torch.maximum(st1, st2)   # [N, M]
    union = torch.maximum(ed1, ed2) - torch.minimum(st1, st2)   # [N, M]
    return inter.clamp(min=0) / union                           # [N, M]


def nms_worker(
    moments: torch.Tensor,      # [M, 2]
    scores1d: torch.Tensor,     # [M]
    threshold: float,
):
    order = scores1d.argsort(descending=True)
    scores1d = scores1d[order]
    moments = moments[order]
    ious = iou(moments, moments)

    try:
        # 5x faster
        from boost import boost_suppression
        suppressed = boost_suppression(ious, threshold)
    except ImportError:
        suppressed = torch.zeros(len(ious), dtype=torch.bool).to(ious.device)
        for i in range(len(ious)):
            if suppressed[i]:
                continue
            suppressed = suppressed | (ious[i] >= threshold)
            suppressed[i] = False

    return moments[~suppressed], scores1d[~suppressed]


def nms(
    moments: Union[List[torch.Tensor], torch.Tensor],                       # [S, M, 2]
    scores1d: Union[List[torch.Tensor], torch.Tensor],                      # [S, M]
    threshold: float,
) -> Tuple[List[torch.tensor], List[torch.tensor]]:
    """batch non-maximum suppression."""
    out_moments = []
    out_scores1ds = []
    num_proposals = []
    for i in range(len(moments)):
        moment, score1d = nms_worker(moments[i], scores1d[i], threshold)
        out_moments.append(moment)
        out_scores1ds.append(score1d)
        num_proposals.append(len(score1d))

    return {
        "out_moments": torch.cat(out_moments),                              # [P, 2]
        "out_scores1ds": torch.cat(out_scores1ds),                          # [P]
        "num_proposals": torch.tensor(num_proposals).to(scores1d.device),  # [S]
    }


def scores2ds_to_moments(
    scores2ds: torch.Tensor,                                            # [B, N, N]
    mask2d: torch.Tensor,                                               # [N, N]
) -> Tuple[torch.Tensor, torch.Tensor]:                                 # [P, 2]
    B, N, _ = scores2ds.shape
    moments = mask2d.nonzero()                                          # [P, 2]
    scores1ds = scores2ds[:, moments[:, 0], moments[:, 1]]              # [B, P]
    moments[:, 1] += 1                                                  # [P, 2]
    moments = moments / N                                               # [P, 2]
    moments = moments.unsqueeze(0).expand(B, -1, -1)                    # [B, P, 2]
    return moments, scores1ds


def moments_to_iou2ds(
    target_moment: torch.Tensor,                                        # [B, 2]
    num_clips: int,                                                     # N = num_clips
) -> torch.Tensor:                                                      # [B, N, N]
    """ Convert batch moment to iou2d."""
    B, _ = target_moment.shape
    moments = target_moment.new_ones(num_clips, num_clips).nonzero()    # [P, 2]
    moments[:, 1] += 1                                                  # [P, 2]
    moments = moments / num_clips                                       # [P, 2]
    iou2d = iou(target_moment, moments)                                 # [B, P]
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
    return torch.stack(feats_bucket, dim=1)


if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)

    num_clips = 16

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
        mask2d = (torch.rand(num_clips, num_clips) > 0.5).triu()
        out_moments, out_scores1ds = scores2ds_to_moments(scores2ds, mask2d)
        pred_moments2 = nms(out_moments, out_scores1ds, threshold=0.3)
