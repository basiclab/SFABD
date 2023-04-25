import torch
import torch.nn as nn
import torch.nn.functional as F


class ProposalConv(nn.Module):
    def __init__(
        self,
        in_channel: int,            # input feature size (512)
        hidden_channel: int,        # hidden feature size (512)
        out_channel: int,           # output feature size (256)
        kernel_size: int,           # kernel size
        num_layers: int,            # number of CNN layers (exclude the projection layers)
        dual_space: bool = False,   # whether to use dual feature scpace
        *args,
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dual_space = dual_space

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

        if dual_space:
            self.proj1 = nn.Conv2d(hidden_channel, out_channel, 1)  # 512 -> 256
            self.proj2 = nn.Conv2d(hidden_channel, out_channel, 1)
        else:
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

        if self.dual_space:
            x1 = self.proj1(x)
            x2 = self.proj2(x)
        else:
            x1 = self.proj(x)
            x2 = x1

        return x1, x2, mask2d


if __name__ == '__main__':
    m = ProposalConv(
        in_channel=512,
        hidden_channel=512,
        out_channel=256,
        kernel_size=9,
        num_layers=4)
    x = torch.randn(4, 512, 64, 64)
    mask = torch.rand(64, 64) > 0.5
    x1, x2, mask = m((x, mask))
    print(x1.shape, x2.shape, mask.shape)

    size = sum([p.numel() for p in m.parameters() if p.requires_grad])
    print(size)
