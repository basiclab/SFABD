from typing import Optional, Callable, Type, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedConv2d(nn.Conv2d):
    def mask_conv(self, mask: Tensor):
        weight = torch.round(F.conv2d(
            mask.float(),
            mask.new_ones(1, 1, self.kernel_size[0], self.kernel_size[1]).float(),
            stride=self.stride,
            padding=self.padding))
        mask = weight > 0
        weight[mask] = 1 / weight[mask]
        return mask, weight

    def forward(self, x: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        if self.kernel_size == (1, 1):
            x = super().forward(x)
        else:
            assert mask is not None
            mask, weight = self.mask_conv(mask)
            x = super().forward(x) * weight
        return x, mask


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> MaskedConv2d:
    """3x3 convolution with padding"""
    return MaskedConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> MaskedConv2d:
    """1x1 convolution"""
    return MaskedConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        identity = x

        out, out_mask = self.conv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, out_mask = self.conv2(out, out_mask)
        out = self.bn2(out)

        if self.downsample is not None:
            identity, identity_mask = self.downsample(x, mask)
            out_mask |= identity_mask

        out += identity
        out = self.relu(out)

        return out, out_mask


class BottleneckBlock(nn.Module):
    expansion: int = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        identity = x

        out, out_mask = self.conv1(x, mask)
        out = self.bn1(out)
        out = self.relu(out)

        out, out_mask = self.conv2(out, out_mask)
        out = self.bn2(out)
        out = self.relu(out)

        out, out_mask = self.conv3(out, out_mask)
        out = self.bn3(out)
        if self.downsample is not None:
            identity, identity_mask = self.downsample(x, mask)
            out_mask |= identity_mask

        out += identity
        out = self.relu(out)

        return out, out_mask


class Downsample(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.conv = conv1x1(in_channel, out_channel, stride)
        self.bn = norm_layer(out_channel)

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        x, mask = self.conv(x, mask)
        x = self.bn(x)
        return x, mask


class MaskedSequential(nn.Sequential):
    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        for module in self:
            x, mask = module(x, mask)
        return x, mask


class MaskedResNet(nn.Module):
    def __init__(
        self,
        in_channel: int,            # input feature size (512)
        hidden_channel: int,        # hidden feature size (512)
        out_channel: int,           # output feature size (256)
        block: Union[Type[BasicBlock], Type[BottleneckBlock]],
        layers: List[int],
    ) -> None:
        super().__init__()
        self.norm_layer = nn.BatchNorm2d

        self.inplanes = in_channel

        self.conv1 = MaskedConv2d(in_channel, hidden_channel, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = self.norm_layer(hidden_channel)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, hidden_channel, layers[0])
        self.layer2 = self._make_layer(block, hidden_channel, layers[1])
        self.layer3 = self._make_layer(block, hidden_channel, layers[2])
        self.layer4 = self._make_layer(block, hidden_channel, layers[3])

    def _make_layer(
        self,
        block: Union[Type[BasicBlock], Type[BottleneckBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,  # always 1 here, do not downsample 2D map
    ) -> MaskedSequential:
        norm_layer = self.norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Downsample(
                self.inplanes, planes * block.expansion, stride, norm_layer)

        layers = []
        # first layer of each stage
        layers.append(
            block(self.inplanes, planes, stride, downsample, norm_layer)
        )
        self.inplanes = planes * block.expansion
        # the remaining layers
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                )
            )

        return MaskedSequential(*layers)

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        x, mask = self.conv1(x, mask)
        x = self.bn1(x)
        x = self.relu(x)
        # x: [bs, hidden, N, N]
        # we don't need downsampling because we want to keep the N x N 2D map
        x, mask = self.layer1(x, mask)
        x, mask = self.layer2(x, mask)
        x, mask = self.layer3(x, mask)
        x, mask = self.layer4(x, mask)
        return x, mask


class ProposalConv(MaskedResNet):
    def __init__(
        self,
        in_channel: int,            # input feature size (512)
        hidden_channel: int,        # hidden feature size (512)
        out_channel: int,           # output feature size (256)
        dual_space: bool = False,   # whether to use dual feature scpace
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channel,
            hidden_channel,
            out_channel,
            # BasicBlock,
            BottleneckBlock,
            # [2, 2, 2, 2]   # resnet-18
            [3, 4, 6, 3]    # resnet-34, or resnet-50 with bottleneck
        )
        self.dual_space = dual_space
        self.block = BottleneckBlock
        # self.block = BasicBlock

        if dual_space:
            self.proj1 = conv1x1(hidden_channel * self.block.expansion, out_channel, 1)  # 512 -> 256
            self.proj2 = conv1x1(hidden_channel * self.block.expansion, out_channel, 1)
        else:
            self.proj = conv1x1(hidden_channel * self.block.expansion, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x, mask2d = input
        mask = mask2d.detach().clone().unsqueeze(0).unsqueeze(0)

        x, _ = super().forward(x, mask)
        if self.dual_space:
            x1, _ = self.proj1(x)
            x2, _ = self.proj2(x)
        else:
            x1, _ = self.proj(x)
            x2 = x1

        return x1, x2, mask2d


if __name__ == '__main__':
    m = ProposalConv(in_channel=512, hidden_channel=512, out_channel=256)
    x = torch.randn(4, 512, 64, 64)
    mask = torch.rand(64, 64) > 0.5
    x1, x2, mask = m((x, mask))
    print(x1.shape, x2.shape, mask.shape)

    size = sum([p.numel() for p in m.parameters() if p.requires_grad])
    print(size)
