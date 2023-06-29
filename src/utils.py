from typing import List, Union, Tuple, Dict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def batch_iou(
    moments1: torch.Tensor,                                     # [B, N, 2]
    moments2: torch.Tensor,                                     # [B, M, 2]
) -> torch.Tensor:                                              # [B, N, M]

    st1 = moments1[:, :, 0:1]                                   # [B, N, 1]
    ed1 = moments1[:, :, 1:2]                                   # [B, N, 1]
    st2 = moments2[:, :, 0:1].permute(0, 2, 1)                  # [B, 1, M]
    ed2 = moments2[:, :, 1:2].permute(0, 2, 1)                  # [B, 1, M]
    inter = torch.minimum(ed1, ed2) - torch.maximum(st1, st2)   # [B, N, M]
    union = torch.maximum(ed1, ed2) - torch.minimum(st1, st2)   # [B, N, M]

    return inter.clamp(min=0) / union                           # [B, N, M]


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
    scores2ds: torch.Tensor,                                            # [S, N, N]
    mask2d: torch.Tensor,                                               # [N, N]
) -> Tuple[torch.Tensor, torch.Tensor]:                                 # [P, 2]
    S, N, _ = scores2ds.shape
    moments = mask2d.nonzero()                                          # [P, 2]
    scores1ds = scores2ds[:, moments[:, 0], moments[:, 1]]              # [S, P]
    moments[:, 1] += 1                                                  # [P, 2]
    moments = moments / N                                               # [P, 2]
    moments = moments.unsqueeze(0).expand(S, -1, -1)                    # [S, P, 2]
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
    # target_moment: [B, 2],   moments: [P, 2]  -> [B, P]
    iou2d = iou(target_moment, moments)                                 # [B, P]
    iou2d = iou2d.view(B, num_clips, num_clips)                         # [B, N, N]
    assert (iou2d >= 0).all() and (iou2d <= 1).all()
    return iou2d


def iou2ds_to_iou2d(
    iou2ds: torch.Tensor,       # [M, N, N]
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


# matplotlib plot
@torch.no_grad()
def plot_moments_on_iou2d(
    iou2d: torch.Tensor,        # [N, N]
    scores2d: torch.Tensor,     # [N, N]
    nms_moments: torch.Tensor,  # [num_proposals_after_nms, 2]
    path: str,
):
    _, N = iou2d.shape  # N: num_clips
    device = iou2d.device
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.5))
    offset = torch.ones(N, N, device=device).triu() * 0.05     # for better visualization
    cm = plt.cm.get_cmap('Reds')

    # plot predicted 2d map score
    scores2d_plot = axs[0].imshow(scores2d + offset, cmap=cm, vmin=0.0, vmax=1.0)
    axs[0].set(xlabel='end index', ylabel='start index')
    axs[0].set_title("score2d")
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(scores2d_plot, cax=cax)

    # not working?
    # plot top 10 nms_moments on both plot
    for i, (row, col) in enumerate(nms_moments[0:10]):
        rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                             fill=False, color='green', linewidth=1.5)
        axs[0].add_patch(rect)

    # plot gt 2d map
    iou2d = iou2d.sub(0.5).div(1.0 - 0.5).clamp(0, 1)   # re-scale iou2d
    gt_plot = axs[1].imshow(iou2d + offset, cmap=cm, vmin=0.0, vmax=1.0)
    axs[1].set(xlabel='end index', ylabel='start index')
    axs[1].set_title(f"GT")
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(gt_plot, cax=cax)

    # plot top 10 nms_moments on both plot
    for i, (row, col) in enumerate(nms_moments[0:10]):
        rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                             fill=False, color='green', linewidth=1.5)
        axs[1].add_patch(rect)

    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)