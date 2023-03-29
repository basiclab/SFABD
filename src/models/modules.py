import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
from transformers import logging


class AggregateVideo(nn.Module):
    def __init__(self, tgt_num: int):
        super().__init__()
        self.tgt_num = tgt_num

    def aggregate_feats(
        self,
        video_feats: torch.Tensor,  # [src_num, C]
    ) -> torch.Tensor:              # [tgt_num, C]
        """Aggregate the feature of video into fixed shape."""
        src_num, _ = video_feats.shape
        idxs = torch.arange(0, self.tgt_num + 1) / self.tgt_num * src_num
        idxs = idxs.round().long().clamp(max=src_num - 1)
        feats_bucket = []
        for i in range(self.tgt_num):
            s, e = idxs[i], idxs[i + 1]
            # print(f"s:{s}, e:{e}")
            if s < e:
                feats_bucket.append(video_feats[s:e].mean(dim=0))
            else:
                feats_bucket.append(video_feats[s])
        return torch.stack(feats_bucket)

    def forward(
        self,
        video_feats: torch.Tensor,          # [B, T, C]
        video_masks: torch.Tensor,          # [B, T]
    ) -> torch.Tensor:                      # [B, tgt_num, C]
        out_feats = []
        for i in range(len(video_feats)):
            out_feat = self.aggregate_feats(video_feats[i][video_masks[i]])
            out_feats.append(out_feat)
        out_feats = torch.stack(out_feats)
        return out_feats


class Conv1dPool(nn.Module):
    def __init__(self, in_channel, out_channel, pool_kernel_size, pool_stride_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(pool_kernel_size, pool_stride_size),
        )

    def forward(
        self,
        x: torch.Tensor         # [B, C, NUM_INIT_CLIPS]
    ):
        """
            B: (B)atch size
            C: (C)hannel = JOINT_SPACE_SIZE
            D: (D)imension of prorposal matrix = NUM_CLIPS
        """
        x = self.model(x)       # [B, C, D]
        return x


class SparseMaxPool(nn.Module):
    def __init__(self, counts):
        super().__init__()
        self.counts = counts
        # self.proj = nn.Conv2d(512, 256, 1)

    def forward(self, x):
        B, C, N = x.shape
        mask2d = torch.eye(N, N, device=x.device).bool()
        x2d = x.new_zeros(B, C, N, N)

        stride, offset = 1, 0
        for level, count in enumerate(self.counts):
            if level != 0:
                x = torch.nn.functional.max_pool1d(x, 3, 2)
            for order in range(count):
                if order != 0:
                    x = torch.nn.functional.max_pool1d(x, 2, 1)
                i = range(0, N - offset, stride)
                j = range(offset, N, stride)
                x2d[:, :, i, j] = x
                mask2d[i, j] = 1
                offset += stride
            offset += stride
            stride *= 2

        return x2d, mask2d


class LanguageModel(nn.Module):
    def __init__(self, joint_space_size, dual_space=False):
        super().__init__()
        self.dual_space = dual_space

        logging.set_verbosity_error()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        logging.set_verbosity_warning()

        if dual_space:
            self.proj1 = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, joint_space_size),
            )
            self.proj2 = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, joint_space_size),
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, joint_space_size),
            )

    def forward(
        self,
        sents_tokens: torch.Tensor,                                         # [S, L]
        sents_masks: torch.Tensor,                                          # [S, L]
    ):
        feats = self.bert(sents_tokens, attention_mask=sents_masks)[0]      # [S, L, C]
        feats = (feats * sents_masks.unsqueeze(-1)).sum(dim=1)              # [S, C]
        feats = feats / sents_masks.sum(dim=1, keepdim=True)                # [S, C]

        if self.dual_space:
            feats1 = self.proj1(feats)                                      # [S, C]
            feats2 = self.proj2(feats)                                      # [S, C]
        else:
            feats1 = self.proj(feats)                                       # [S, C]
            feats2 = feats1                                                 # [S, C]
        return feats1, feats2
