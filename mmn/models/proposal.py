import torch
import torch.nn as nn
import torch.nn.functional as F


class ProposalConv(nn.Module):
    def __init__(
        self,
        in_dim: int,        # number of clips
        in_channel: int,    # input feature size
        hidden_channel: int,   # hidden feature size
        out_channel: int,   # output feature size
        kernel_size: int,   # kernel size
        num_layers: int,    # number of CNN layers (exclude the last projection layer)
    ):
        super(ProposalConv, self).__init__()
        # Padding to ensure the dimension of the output map2d

        # [1, 1, 1, ..., 1]
        # [0, 1, 1, ..., 1]
        # [0, 0, 1, ..., 1]
        # ...
        # [0, 0, 0, ..., 1]
        mask = torch.triu(torch.ones(1, 1, in_dim, in_dim)).float()

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            if idx == 0:
                padding = (kernel_size - 1) * num_layers // 2
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(
                        in_channel, hidden_channel, kernel_size, padding=padding),
                    nn.BatchNorm2d(hidden_channel),
                    nn.ReLU(inplace=False),
                ))
                mask = self.conv_mask(mask, kernel_size, 1, padding)
            else:
                self.blocks.append(nn.Sequential(
                    nn.Conv2d(hidden_channel, hidden_channel, kernel_size),
                    nn.BatchNorm2d(hidden_channel),
                    nn.ReLU(inplace=False),
                ))
                mask = self.conv_mask(mask, kernel_size, 1, padding=0)
            self.register_buffer(f'mask{idx}', mask)

        self.proj = nn.Conv2d(hidden_channel, out_channel, 1)

    def conv_mask(self, mask, kernel_size, stride, padding):
        weight = torch.ones(1, 1, kernel_size, kernel_size)
        next_mask = F.conv2d(mask, weight, stride=stride, padding=padding)
        next_mask = torch.round(next_mask)                      # TODO: magic
        return next_mask

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            mask = getattr(self, f'mask{idx}')
            x = block(x) * mask
        x = self.proj(x)
        return x
