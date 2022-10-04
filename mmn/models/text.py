import torch
import torch.nn as nn
from transformers import DistilBertModel
from transformers import logging


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
