import torch
from torch import nn


class FeatAvgPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super(FeatAvgPool, self).__init__()
        ## ex. (64, 4096) -> (64, 512)
        self.conv = nn.Conv1d(input_size, hidden_size, 1, 1) ## input_ch, output_ch, kernel_size, stride
        ## kernel size = 2, stride = num_pre_clips // num_clips  64 // 32 = 2
        ## ex. (64, 512) -> (32, 512)
        self.pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, x):
        x = x.transpose(1, 2)  # B, C, T
        return self.pool(self.conv(x).relu())

def build_featpool(cfg):
    input_size = cfg.MODEL.MMN.FEATPOOL.INPUT_SIZE ## 4096
    hidden_size = cfg.MODEL.MMN.FEATPOOL.HIDDEN_SIZE ## 512
    kernel_size = cfg.MODEL.MMN.FEATPOOL.KERNEL_SIZE  # 2 for charades
    stride = cfg.INPUT.NUM_PRE_CLIPS // cfg.MODEL.MMN.NUM_CLIPS ## 32 // 16 = 2 -> 64 / 32 = 2
    return FeatAvgPool(input_size, hidden_size, kernel_size, stride)
