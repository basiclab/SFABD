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


class ProposalPool(nn.Module):
    def forward(self, x: torch.Tensor):
        """
            B: (B)atch size
            C: (C)hannel = JOINT_SPACE_SIZE
            D: (D)imension of prorposal matrix = NUM_CLIPS
        """
        B, C, D = x.shape
        zero = x.new_zeros(B, C)
        x2d = []
        for i in range(D):
            for j in range(D):
                if i <= j:
                    x2d.append(x[:, :, i: j + 1].max(dim=2).values)
                else:
                    x2d.append(zero)

        x2d = torch.stack(x2d, dim=-1).view(B, C, D, D)
        return x2d


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
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(
                        in_channel, hidden_channel, kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_channel),
                    nn.ReLU(inplace=True),
                ))
                self.paddings.append(padding)
            else:
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(hidden_channel, hidden_channel, kernel_size),
                    nn.BatchNorm2d(hidden_channel),
                    nn.ReLU(inplace=True),
                ))
                self.paddings.append(0)

        self.proj = nn.Conv2d(hidden_channel, out_channel, 1)

    def conv_mask(self, mask, padding):
        masked_weight = torch.round(F.conv2d(
            mask.float(),
            mask.new_ones(1, 1, self.kernel_size, self.kernel_size).float(),
            padding=padding))
        masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0]
        mask = masked_weight > 0
        return mask, masked_weight

    def forward(self, x):
        _, _, D, _ = x.shape
        mask = x.new_ones(1, 1, D, D).bool().triu()
        for padding, block in zip(self.paddings, self.blocks):
            mask, masked_weight = self.conv_mask(mask, padding)
            x = block(x) * masked_weight
        x = self.proj(x)
        return x


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
        length: torch.Tensor,             # [N]
    ):
        attention_mask = torch.arange(tokens.size(1), device=tokens.device)     # [T]
        attention_mask = attention_mask.repeat(tokens.size(0), 1)               # [N, T]
        attention_mask = attention_mask < length.unsqueeze(-1)                    # [N, T]

        feats = self.bert(tokens, attention_mask=attention_mask)[0]             # [N, T, C]
        feats = (feats * attention_mask.unsqueeze(-1)).sum(dim=1)               # [N, C]
        feats = feats / length.unsqueeze(-1)                                      # [N, C]

        feats = self.proj(feats)                                                # [N, C]
        return feats
