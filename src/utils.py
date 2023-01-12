from typing import List, Union, Tuple, Dict

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
) -> Dict[str, torch.Tensor]:
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
        "num_proposals": torch.tensor(num_proposals).to(scores1d.device),   # [S]
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


def l2_normalize(tensor, axis=-1):
    """L2-normalize columns of tensor"""
    return F.normalize(tensor, p=2, dim=axis)


## last dim must be hidden_size C
def sample_gaussian_tensors(
    mu: torch.Tensor,       # [..., C] 
    logsigma: torch.Tensor, # [..., C]
    num_samples: int=7,     
) -> torch.Tensor:          # [..., num_samples, C]
    repeat_shape_list = [1 for i in range(len(mu.shape)+1)]     # [1, 1, ... 1, 1]
    repeat_shape_list[len(repeat_shape_list)-2] = num_samples   # [1, 1, ... num_samples, 1]  
    new_shape = torch.zeros_like(mu).unsqueeze(-2).repeat(repeat_shape_list).shape      ## [.., num_samples, C]
    sampled_normal_vector = torch.randn(new_shape, dtype=mu.dtype, device=mu.device)    ## [.., num_samples, C]
    sampled_feats = sampled_normal_vector.mul(torch.exp(logsigma.unsqueeze(-2))).add_(mu.unsqueeze(-2))
    return sampled_feats  ## not normalized


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
