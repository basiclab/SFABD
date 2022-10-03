from typing import List, Tuple

import h5py
import torch
import torch.nn.functional as F


def iou(
    target: torch.Tensor,           # [2]
    moments: torch.Tensor           # [..., 2]
) -> torch.Tensor:
    """
    Returns:
        iou: shape = [...]
    """
    moments_start, moments_end = moments[..., 0], moments[..., 1]
    start, end = target
    inter = moments_end.min(end) - moments_start.max(start)
    union = moments_end.max(end) - moments_start.min(start)
    return inter.clamp(min=0) / union


def nms(
    moments: torch.Tensor,          # [?, 2]
    scores1d: torch.Tensor,         # [?]
    threshold: float,
) -> torch.Tensor:                  # [?]
    """Non-maximum suppression.
    Returns:
        mask: [?], True for the moments that are not suppressed
    """
    N = moments.shape[0]
    ranks = scores1d.argsort(descending=True)
    mask = scores1d.new_ones(N, dtype=torch.bool)
    for idx in ranks:           # iterate from highest score to lowest score
        if mask[idx]:
            mask[iou(moments[idx], moments) > threshold] = False
            mask[idx] = True
    return mask


def scores2d_to_moments_scores1d(
    scores2d: torch.Tensor,         # [D, D] D = NUM_CLIPS
    duration: float,                # video length in seconds
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the moments and scores of upper triangle of scores2d.

    Returns:
        moments: [?, 2] in seconds
        scores1d: [?]
    """
    D, _ = scores2d.shape
    device = scores2d.device
    moments = torch.ones(D, D, device=device).triu().nonzero()
    scores1d = scores2d[moments[:, 0], moments[:, 1]]
    moments[:, 1] += 1
    moments = moments * duration / D
    return moments, scores1d


# moment: ex. [4.0, 8.4]
def moment_to_iou2d(
    target_moment: torch.Tensor,    # [2]
    num_clips: int,
    duration: float
) -> torch.Tensor:
    """
    Returns:
        iou2d: [D, D]
    """
    moments = target_moment.new_ones(num_clips, num_clips).nonzero()
    moments[:, 1] += 1
    moments = moments * duration / num_clips
    iou2d = iou(target_moment, moments).reshape(num_clips, num_clips)
    return iou2d


def moments_to_iou2d(
    target_moments: torch.Tensor,   # [?, 2]
    num_clips: int,
    duration: float
):
    """
    Step 1. generate iou2d for each label
    Step 2. iterate each iou2d map, substract the rest iou2d maps from it,
            clamp negative value, and only keep clips that > 0.5 (make the
            combined iou map has clean boundary)
    Returns:
        iou2d: [D, D]
    """
    iou2ds = []
    for target_moment in target_moments:
        iou2d = moment_to_iou2d(target_moment, num_clips, duration)
        iou2ds.append(iou2d)
    iou2ds = torch.stack(iou2ds)                    # [?, D, D]
    iou2d_sum = iou2ds.sum(dim=0, keepdim=True)     # [D, D]
    iou2ds = iou2ds - (iou2d_sum - iou2ds)          # [?, D, D]
    iou2ds = iou2ds * (iou2ds > 0.5)                # [?, D, D]
    iou2d = iou2ds.sum(dim=0)                       # [D, D]
    return iou2d


def vgg_feats(
    feat_file: str,                             # path to h5 file
    vid: str,                                   # video id
    num_init_clips: int,                        # target dimension
):
    with h5py.File(feat_file, 'r') as f:
        feats = f[vid][:]
        feats = torch.from_numpy(feats).float()
    return aggregate_feats(feats, num_init_clips, op_type='avg')


def c3d_feats(
    feat_folder: str,                           # path to feature folder
    vid: str,                                   # video id
    num_init_clips: int,                        # target dimension
) -> torch.Tensor:
    feats = torch.load(f"{feat_folder}/{vid}.pt")

    return aggregate_feats(feats, num_init_clips, op_type='avg')


def multi_vgg_feats(
    feat_file: str,                             # path to h5 file
    vids: List[str],                            # list of video ids
    seq_timestamps: List[Tuple[float, float]],  # list of (start, end) timestamps
    num_init_clips: int,                        # target dimension
) -> torch.Tensor:
    assert len(vids) == len(seq_timestamps)

    with h5py.File(feat_file, 'r') as f:
        feats = []
        for vid, (seq_start, seq_end) in zip(vids, seq_timestamps):
            feat = f[vid][seq_start: seq_end]
            feats.append(torch.from_numpy(feat).float())
        feats = torch.cat(feats, dim=0)

    return aggregate_feats(feats, num_init_clips, op_type='avg')


def multi_c3d_feats(
    feat_folder: str,                           # path to feature folder
    vids: List[int],                            # list of video ids
    seq_timestamps: List[Tuple[float, float]],  # list of (start, end) timestamps
    num_init_clips: int,                        # target dimension
) -> torch.Tensor:
    assert len(vids) == len(seq_timestamps)

    feats = []
    for vid, (seq_start, seq_end) in zip(vids, seq_timestamps):
        feat = torch.load(f"{feat_folder}/{vid}.pt")
        feat = feat[seq_start:seq_end]
        feats.append(feat)
    feats = torch.cat(feats, dim=0)

    return aggregate_feats(feats, num_init_clips, op_type='avg')


def aggregate_feats(
    feats: torch.Tensor,                        # [NUM_SRC_CLIPS, C]
    num_tgt_clips: int,                         # number of target clip
    op_type: str = 'avg',                       # 'avg' or 'max'
) -> torch.Tensor:
    """Produce the feature of per video into fixed shape by averaging.

    Returns:
        avgfeats: [C, num_tgt_clips]
    """
    assert op_type in ['avg', 'max']

    num_src_clips, _ = feats.shape
    idxs = torch.arange(0, num_tgt_clips + 1) / num_tgt_clips * num_src_clips
    idxs = idxs.round().long().clamp(max=num_src_clips - 1)
    feats = F.normalize(feats, dim=1)
    feats_bucket = []
    for i in range(num_tgt_clips):
        s, e = idxs[i], idxs[i + 1]
        # To prevent an empty selection, check the indices are valid.
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
    # test
    target_moment = torch.tensor([1.5, 2.5])
    num_clips = 5
    duration = 10.0

    # test `moment_to_iou2d`
    iou2d = moment_to_iou2d(target_moment, num_clips, duration)
    assert iou2d[0, 1] == 1 / 4, f"{iou2d[0, 1]}"       # 0.25
    assert iou2d[1, 1] == 0.5 / 2.5, f"{iou2d[1, 1]}"   # 0.2

    # test `scores2d_to_moments_scores1d`
    moments, scores1d = scores2d_to_moments_scores1d(iou2d, duration)
    assert (moments[0].numpy() == [0., 2.]).all(), f"{moments[0]}"
    assert (moments[4].numpy() == [0., 10.]).all(), f"{moments[4]}"
    assert (moments[10].numpy() == [4., 8.]).all(), f"{moments[10]}"

    # test `nms`
    mask = nms(moments, scores1d, threshold=0.1)
    assert mask.int().sum().item() == 4, f"{mask.int().sum().item()}"
    assert mask[1].item() is True, f"{mask[1]}"
    assert mask[9].item() is True, f"{mask[9]}"
    assert mask[12].item() is True, f"{mask[12]}"
    assert mask[14].item() is True, f"{mask[14]}"

    # test `moment_to_iou2d`
    target_moments = torch.tensor([[0, 3], [4, 6], [8, 10]])
    num_clips = 8
    duration = 10.0
    iou2d = moments_to_iou2d(target_moments, num_clips, duration)
    assert iou2d[0, 1] == 2.5 / 3, f"{iou2d[0, 1]}"     # 0.8333
    assert iou2d[3, 4] == 2 / 2.5, f"{iou2d[1, 1]}"     # 0.8000
    assert iou2d[6, 7] == 2 / 2.5, f"{iou2d[1, 1]}"     # 0.8000
