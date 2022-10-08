from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import (
    Conv1dPool, SparseMaxPool, ProposalConv, LanguageModel)


class MMN(nn.Module):
    def __init__(
        self,
        feat1d_in_channel: int,                 # Conv1dPool
        feat1d_out_channel: int = 512,          # Conv1dPool
        feat1d_pool_kerenl_size: int = 2,       # Conv1dPool
        feat1d_pool_stride_size: int = 2,       # Conv1dPool
        feat2d_pool_counts: List[int] = [16],   # SparseMaxPool
        conv2d_hidden_channel: int = 512,       # ProposalConv
        conv2d_kernel_size: int = 5,            # ProposalConv
        conv2d_num_layers: int = 8,             # ProposalConv
        joint_space_size: int = 256,
    ):
        """
            B: (B)atch size
            C: (C)hannel = JOINT_SPACE_SIZE
            N: (N)um clips
        """
        super(MMN, self).__init__()
        self.video_model = nn.Sequential(
            Conv1dPool(                                     # [B, C, NUM_INIT_CLIPS]
                feat1d_in_channel,
                feat1d_out_channel,
                feat1d_pool_kerenl_size,
                feat1d_pool_stride_size,
            ),                                              # [B, C, N]
            SparseMaxPool(feat2d_pool_counts),              # [B, C, N, N]
            ProposalConv(
                feat1d_out_channel,
                conv2d_hidden_channel,
                joint_space_size,
                conv2d_kernel_size,
                conv2d_num_layers,
            )                                               # [B, C, N, N]
        )
        self.query_model = LanguageModel(joint_space_size)  # [S, C]

    def matching_scores(
        self,
        video_feats: torch.Tensor,  # [B, C, N, N]
        query_feats: torch.Tensor,  # [S, C]
        scatter_idx: torch.Tensor,  # [S]
        mask2d: torch.Tensor,       # [N, N]
    ):
        """
        Return matching scores between each proposal and corresponding query.

        Return:
            scores_matrix: [B, N, N], the value range is [-1, 1]
        """
        video_feats = F.normalize(video_feats, dim=1)
        query_feats = F.normalize(query_feats, dim=1)
        scores2d = video_feats[scatter_idx] * query_feats[:, :, None, None]
        scores2d = scores2d.sum(dim=1)              # [S, N, N]
        scores2d = torch.sigmoid(scores2d * 10)     # [S, N, N]
        scores2d = scores2d * mask2d.unsqueeze(0)   # [S, N, N]
        return scores2d

    def forward(
        self,
        video_feats: torch.Tensor,          # [B, C, NUM_INIT_CLIPS]
        sents_tokens: torch.Tensor,         # [S, L]
        sents_length: torch.Tensor,         # [S]
        num_targets: torch.Tensor,          # [B]
        **kwargs,                           # dummy
    ):
        """
            B: (B)atch size
            C: (C)hannel = JOINT_SPACE_SIZE
            N: (N)um clips
            S: number of (S)entences
            L: (L)ength of tokens
        """
        B, = num_targets.shape
        device = num_targets.device
        scatter_idx = torch.arange(B).to(device).repeat_interleave(num_targets)
        video_feats, mask2d = self.video_model(video_feats)
        sents_feats = self.query_model(sents_tokens, sents_length)
        scores2d = self.matching_scores(
            video_feats, sents_feats, scatter_idx, mask2d)

        return video_feats, sents_feats, scores2d, mask2d


if __name__ == '__main__':
    from transformers import DistilBertTokenizer

    def test(
        B=8,
        INIT_CHANNEL=4096,
        NUM_INIT_CLIPS=32,
        NUM_CLIPS=16,
        feat1d_out_channel=512,
        feat1d_pool_kerenl_size=2,
        feat2d_pool_counts=[16],
        conv2d_hidden_channel=512,
        conv2d_kernel_size=5,
        conv2d_num_layers=8,
        joint_space_size=256,
    ):
        model = MMN(
            feat1d_in_channel=INIT_CHANNEL,
            feat1d_out_channel=feat1d_out_channel,
            feat1d_pool_kerenl_size=feat1d_pool_kerenl_size,
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
        num_targets = torch.tensor([1, 1, 2, 2, 3, 1, 3, 2])
        S = sum(num_targets)
        assert S == len(sents)
        sents = tokenizer(
            sents, padding=True, return_tensors="pt", return_length=True)

        video_feats = torch.randn(B, INIT_CHANNEL, NUM_INIT_CLIPS)
        sents_tokens = sents['input_ids']
        sents_lengthgth = sents['length']

        video_feats, sents_feats, scores, mask2d = model(
            video_feats, sents_tokens, sents_lengthgth, num_targets)

        print('-' * 80)
        print(f"video_feats   : {video_feats.shape}")
        print(f"sentence_feats: {sents_feats.shape}")
        print(f"scores        : {scores.shape}")
        print(f"mask2d        : {mask2d.shape}")

        assert video_feats.shape == (B, joint_space_size, NUM_CLIPS, NUM_CLIPS)
        assert sents_feats.shape == (S, joint_space_size)
        assert scores.shape == (sum(num_targets).item(), NUM_CLIPS, NUM_CLIPS)
        assert mask2d.shape == (NUM_CLIPS, NUM_CLIPS)

    # Charades
    test(
        INIT_CHANNEL=4096,
        NUM_INIT_CLIPS=32,
        NUM_CLIPS=16,
        feat1d_out_channel=512,
        feat1d_pool_kerenl_size=2,
        feat2d_pool_counts=[16],
        conv2d_hidden_channel=512,
        conv2d_kernel_size=5,
        conv2d_num_layers=8,
        joint_space_size=256,
    )

    # ActivityNet
    test(
        INIT_CHANNEL=500,
        NUM_INIT_CLIPS=256,
        NUM_CLIPS=64,
        feat1d_out_channel=512,
        feat1d_pool_kerenl_size=4,
        feat2d_pool_counts=[16, 8, 8],
        conv2d_hidden_channel=512,
        conv2d_kernel_size=9,
        conv2d_num_layers=4,
        joint_space_size=256,
    )

    # TACoS
    test(
        INIT_CHANNEL=4096,
        NUM_INIT_CLIPS=256,
        NUM_CLIPS=128,
        feat1d_out_channel=512,
        feat1d_pool_kerenl_size=2,
        feat2d_pool_counts=[16, 8, 8, 8],
        conv2d_hidden_channel=512,
        conv2d_kernel_size=5,
        conv2d_num_layers=8,
        joint_space_size=256,
    )
