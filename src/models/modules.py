import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
from transformers import logging


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


class ProposalConv(nn.Module):
    def __init__(
        self,
        in_channel: int,        # input feature size
        hidden_channel: int,    # hidden feature size
        out_channel: int,       # output feature size
        kernel_size: int,       # kernel size
        num_layers: int,        # number of CNN layers (exclude the last projection layer)
    ):
        super(ProposalConv, self).__init__()
        self.kernel_size = kernel_size

        self.blocks = nn.ModuleList()
        self.paddings = []
        for idx in range(num_layers):
            if idx == 0:
                padding = (kernel_size - 1) * num_layers // 2
                channel = in_channel
            else:
                padding = 0
                channel = hidden_channel
            self.blocks.append(nn.Sequential(
                nn.Conv2d(
                    channel, hidden_channel, kernel_size, padding=padding),
                nn.BatchNorm2d(hidden_channel),
                nn.ReLU(inplace=True),
            ))
            self.paddings.append(padding)

        self.proj = nn.Conv2d(hidden_channel, out_channel, 1)

    def get_masked_weight(self, mask, padding):
        masked_weight = torch.round(F.conv2d(
            mask.float(),
            mask.new_ones(1, 1, self.kernel_size, self.kernel_size).float(),
            padding=padding))
        masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0]
        mask = masked_weight > 0
        return mask, masked_weight

    def forward(self, input):
        x, mask2d = input
        mask = mask2d.detach().clone().unsqueeze(0).unsqueeze(0)
        for padding, block in zip(self.paddings, self.blocks):
            mask, masked_weight = self.get_masked_weight(mask, padding)
            x = block(x) * masked_weight
        x = self.proj(x)
        return x, mask2d


class LanguageModel(nn.Module):
    def __init__(self, joint_space_size):
        super().__init__()
        logging.set_verbosity_error()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        logging.set_verbosity_warning()
        self.proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, joint_space_size),
        )

    def forward(
        self,
        tokens: torch.Tensor,           # [N, T]
        length: torch.Tensor,           # [N]
    ):
        attention_mask = torch.arange(tokens.size(1), device=tokens.device)     # [T]
        attention_mask = attention_mask.repeat(tokens.size(0), 1)               # [N, T]
        attention_mask = attention_mask < length.unsqueeze(-1)                    # [N, T]

        feats = self.bert(tokens, attention_mask=attention_mask)[0]             # [N, T, C]
        feats = (feats * attention_mask.unsqueeze(-1)).sum(dim=1)               # [N, C]
        feats = feats / length.unsqueeze(-1)                                      # [N, C]

        feats = self.proj(feats)                                                # [N, C]
        return feats
