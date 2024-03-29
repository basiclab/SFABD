from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.misc import construct_class
from src.models.modules import (
    AggregateVideo, Conv1dPool, SparseMaxPool, LanguageModel)


# cos sim between 2D proposal map and query
def compute_scores(
    video_feats: torch.Tensor,      # [S, C, N, N]
    sents_feats: torch.Tensor,      # [S, C]
) -> torch.Tensor:                  # [S, N, N]
    """
    Return cosine similarity between each proposal and corresponding query.

    Return:
        scores_matrix: [S, N, N], the value range is [-1, 1]
    """
    video_feats = F.normalize(video_feats, dim=1)
    sents_feats = F.normalize(sents_feats, dim=1)
    # cosine sim
    scores2d = video_feats * sents_feats[:, :, None, None]
    scores2d = scores2d.sum(dim=1)   # [S, N, N]
    return scores2d


def iou_scores(
    video_feats: torch.Tensor,              # [B, C, N, N]
    sents_feats: torch.Tensor,              # [S, C]
    mask2d: torch.Tensor,                   # [N, N]
    scale: float = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:     # [S, N, N]
    """
    Return matching scores between each proposal and corresponding query.

    Return:
        scores2d: the value range is [0, 1]
        logits2d: the value range is [-scale, +scale]
    """
    scores2d = compute_scores(video_feats, sents_feats)  # [-1, 1]
    logits2d = scores2d * scale                                         # [-scale, scale]
    scores2d = torch.sigmoid(logits2d.detach())                         # [0, 1]
    scores2d = scores2d * mask2d.unsqueeze(0)
    return scores2d, logits2d


def con_scores(
    video_feats: torch.Tensor,      # [B, C, N, N]
    sents_feats: torch.Tensor,      # [S, C]
    mask2d: torch.Tensor,           # [N, N]
) -> torch.Tensor:                  # [S, N, N]
    """
    Return matching scores between each proposal and corresponding query.

    Return:
        scores_matrix: [B, N, N], the value range is [0, 1]
    """
    scores2d = compute_scores(video_feats, sents_feats)
    scores2d = (scores2d + 1) / 2
    scores2d = scores2d * mask2d.unsqueeze(0)
    return scores2d


class MMN(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_init_clips: int,
        feat1d_in_channel: int,                 # Conv1dPool
        feat1d_out_channel: int = 512,          # Conv1dPool
        feat1d_pool_kernel_size: int = 2,       # Conv1dPool
        feat1d_pool_stride_size: int = 2,       # Conv1dPool
        feat2d_pool_counts: List[int] = [16],   # SparseMaxPool
        conv2d_hidden_channel: int = 512,       # ProposalConv
        conv2d_kernel_size: int = 5,            # ProposalConv
        conv2d_num_layers: int = 8,             # ProposalConv
        joint_space_size: int = 256,
        resnet: int = 18,
        dual_space: bool = False,               # whether to use dual feature space
    ):
        """
            B: (B)atch size
            C: (C)hannel = JOINT_SPACE_SIZE
            N: (N)um clips
        """
        super(MMN, self).__init__()
        self.dual_space = dual_space
        self.aggregate = AggregateVideo(num_init_clips)

        if "resnet" in backbone:
            self.video_model = nn.Sequential(
                Conv1dPool(                                     # [B, C, NUM_INIT_CLIPS]
                    feat1d_in_channel,
                    feat1d_out_channel,
                    feat1d_pool_kernel_size,
                    feat1d_pool_stride_size,
                ),                                              # [B, C, N]
                SparseMaxPool(feat2d_pool_counts),              # [B, C, N, N]
                construct_class(
                    backbone,
                    in_channel=feat1d_out_channel,
                    hidden_channel=conv2d_hidden_channel,
                    out_channel=joint_space_size,
                    kernel_size=conv2d_kernel_size,
                    num_layers=conv2d_num_layers,
                    resnet=resnet,
                    dual_space=dual_space,                      # [B, C, N, N]
                ),
            )
        # ConvNet in MMN paper
        else:
            self.video_model = nn.Sequential(
                Conv1dPool(                                     # [B, C, NUM_INIT_CLIPS]
                    feat1d_in_channel,
                    feat1d_out_channel,
                    feat1d_pool_kernel_size,
                    feat1d_pool_stride_size,
                ),                                              # [B, C, N]
                SparseMaxPool(feat2d_pool_counts),              # [B, C, N, N]
                construct_class(
                    backbone,
                    in_channel=feat1d_out_channel,
                    hidden_channel=conv2d_hidden_channel,
                    out_channel=joint_space_size,
                    kernel_size=conv2d_kernel_size,
                    num_layers=conv2d_num_layers,
                    dual_space=dual_space,                      # [B, C, N, N]
                ),
            )

        self.sents_model = LanguageModel(joint_space_size, dual_space)                   # [S, C]

    def forward(
        self,
        video_feats: torch.Tensor,          # [B, T, C]
        video_masks: torch.Tensor,          # [B, T]
        sents_tokens: torch.Tensor,         # [S, L]
        sents_masks: torch.Tensor,          # [S, L]
        num_sentences: torch.Tensor,        # [B]
        **kwargs,                           # dummy
    ):
        """
            B: (B)atch size
            C: (C)hannel = JOINT_SPACE_SIZE
            N: (N)um clips
            S: number of (S)entences
            L: (L)ength of tokens
        """
        assert sents_tokens.shape == sents_masks.shape
        assert sents_tokens.shape[0] == num_sentences.sum()

        B = video_feats.shape[0]

        video_feats = self.aggregate(video_feats, video_masks)      # [S, ?, C]
        video_feats = video_feats.permute(0, 2, 1)                  # [S, C, ?]
        video_feats1, video_feats2, mask2d = self.video_model(video_feats)
        sents_feats1, sents_feats2 = self.sents_model(
            sents_tokens,
            sents_masks
        )                                                           # [S, C]

        if self.dual_space:
            iou_scores2d, logits2d = iou_scores(
                video_feats1, sents_feats1, mask2d)
            con_scores2d = con_scores(
                video_feats2, sents_feats2, mask2d)
            scores2d = torch.sqrt(con_scores2d) * iou_scores2d
        else:
            scores2d, logits2d = iou_scores(
                video_feats1, sents_feats1, mask2d)

        return (
            video_feats2,       # [S, C, N, N]  for contrastive learning
            sents_feats2,       # [S, C]        for contrastive learning
            logits2d,           # [S, N, N]     for iou loss (sim score * scale=10)
            scores2d.detach(),  # [S, N, N]     for evaluation
            mask2d.detach(),    # [N, N]
        )


if __name__ == '__main__':
    from transformers import DistilBertTokenizer

    def test(
        B=8,
        INIT_CHANNEL=2816,
        NUM_INIT_CLIPS=128,
        NUM_CLIPS=64,
        feat1d_out_channel=512,
        feat1d_pool_kernel_size=2,
        feat2d_pool_counts=[16, 8, 8],
        conv2d_hidden_channel=512,
        conv2d_kernel_size=9,
        conv2d_num_layers=4,
        joint_space_size=256,
    ):
        model = MMN(
            num_init_clips=NUM_INIT_CLIPS,
            feat1d_in_channel=INIT_CHANNEL,
            feat1d_out_channel=feat1d_out_channel,
            feat1d_pool_kernel_size=feat1d_pool_kernel_size,
            feat1d_pool_stride_size=NUM_INIT_CLIPS // NUM_CLIPS,
            feat2d_pool_counts=feat2d_pool_counts,
            conv2d_hidden_channel=conv2d_hidden_channel,
            conv2d_kernel_size=conv2d_kernel_size,
            conv2d_num_layers=conv2d_num_layers,
            joint_space_size=joint_space_size
        )

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        sents = [
            "a person is putting a book on a shelf.",
            "person begins to play on a phone.",
            "person closes the door behind them.",
            "the person closes the dryer door.",
            "person closes the frig door.",
            "a person closes the door.",
            "a person is looking out a window.",
            "person standing to look out the window.",
            "another person is looking out a window.",
            "person eats the sandwich.",
            "a person is sitting at at table eating a sandwich.",
            "person eating something.",
            "person eating something.",
            "person sitting on the floor.",
            "a person sitting in a chair takes off their shoes.",
        ]
        num_sentences = torch.tensor([1, 1, 2, 2, 3, 1, 3, 2])
        S = sum(num_sentences)
        assert S == len(sents)
        sents = tokenizer(sents, padding=True, return_tensors="pt")

        pad_len = 2 * NUM_INIT_CLIPS + 5
        video_feats = torch.randn(B, pad_len, INIT_CHANNEL)
        video_lens = torch.randint(NUM_INIT_CLIPS - 5, pad_len, (B,))
        video_masks = torch.arange(pad_len)[None, :] < video_lens[:, None]
        sents_tokens = sents['input_ids']
        sents_masks = sents['attention_mask']

        video_feats, sents_feats, scores2d, logits2d, mask2d = model(
            video_feats, video_masks, sents_tokens, sents_masks, num_sentences)

        print('-' * 80)
        print(f"video_feats   : {video_feats.shape}")
        print(f"sentence_feats: {sents_feats.shape}")
        print(f"scores2d      : {scores2d.shape}")
        print(f"logits2d      : {logits2d.shape}")
        print(f"mask2d        : {mask2d.shape}")

        assert video_feats.shape == (B, joint_space_size, NUM_CLIPS, NUM_CLIPS)
        assert sents_feats.shape == (S, joint_space_size)
        assert scores2d.shape == (sum(num_sentences).item(), NUM_CLIPS, NUM_CLIPS)
        assert logits2d.shape == (sum(num_sentences).item(), NUM_CLIPS, NUM_CLIPS)
        assert mask2d.shape == (NUM_CLIPS, NUM_CLIPS)

    # QVhighlights
    test(
        INIT_CHANNEL=2816,              # 2304 + 512
        NUM_INIT_CLIPS=128,
        NUM_CLIPS=64,
        feat1d_out_channel=512,
        feat1d_pool_kernel_size=2,
        feat2d_pool_counts=[16, 8, 8],
        conv2d_hidden_channel=512,
        conv2d_kernel_size=9,
        conv2d_num_layers=4,
        joint_space_size=256,
    )
