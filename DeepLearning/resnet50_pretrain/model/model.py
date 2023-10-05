from typing import Optional, List

import torch.nn as nn
from torch import Tensor, flatten

__all__ = ["ResNet50"]


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Residual Block for resnet50
    expansion: int = 4

    def __init__(
            self,
            in_channels: int,  # the number of input channels of building block
            out_channels: int,  # the number of channels of the first line of building block,
            # note in resnet50, kaiming set the number of channels of last line of each building block as four times
            # of this parameter, this is the role of expansion.
            stride: int = 1,
            padding: int = 1,
            downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(
            self,
            layers: List[int] = [3, 4, 6, 3],
            num_classes: int = 1000,
    ) -> None:
        super().__init__()
        # this parameter indicates the number of channels received by the residual layer.
        # it will be updated when building different residual block.
        self.in_channels = 64

        # the size of input images is 224 × 224
        # o = ⌊(n + 2p - f) / s⌋ + 1 = ⌊(224 + 2 × 3 - 7) / 2⌋ + 1 = 112,
        # thus current shape of feature map is 112 × 112 × self.in_channels.
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # o = ⌊(n + 2p - f) / s⌋ + 1 = ⌊(112 + 2 × 1 - 3) / 2⌋ + 1 = 56,
        # thus we get 56 × 56 × self.out_channels.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(
            self,
            channels: int,  # this parameter indicates that the number of channels of guy which is output by
            # the first line in each building block.
            num_blocks: int,  # the parameter indicates the number of blocks possessed by layer.
            stride: int = 1,  # the parameter controlling whether apply down sampling operation when forward.
    ) -> nn.Sequential:
        # stride != 1 means that the length and width of feature map is not coincide. It will take effect in conv3_x,
        # conv4_x and conv5_x illustrated in Table1 of original paper.
        # apart from above, self.in_channels != channels * Bottleneck.expansion means that the depth of feature map is
        # not coincide. in resnet50, this case will happen in the conv2_x stated in Table1 of original paper.
        if stride != 1 or self.in_channels != channels * Bottleneck.expansion:
            # down sampling is aim to make the shape of input coincides with output of last line in residual block.
            downsample = nn.Sequential(
                conv1x1(self.in_channels, channels * Bottleneck.expansion, stride),
                nn.BatchNorm2d(channels * Bottleneck.expansion),
            )
        else:
            downsample = None

        layers = []
        # There are two reasons for adding the first building block individually to the layer
        # 1. Down sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2.
        # 2. In conv2_x. conv3_x. conv4_x needs to pass in stried=2 on the first block to halve the length and width.
        layers.append(
            Bottleneck(
                self.in_channels, channels, stride=stride, downsample=downsample,
            )
        )
        # update in_channels to build latter residual block
        self.in_channels = channels * Bottleneck.expansion

        for _ in range(1, num_blocks):
            layers.append(
                Bottleneck(
                    self.in_channels,
                    channels,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        pass
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = flatten(out, 1)
        out = self.fc(out)

        return out
