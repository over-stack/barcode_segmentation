import torch
from torch import nn
import torch.nn.functional as F
import math


class Conv2DSamePad(torch.nn.Conv2d):
    @staticmethod
    def calc_same_pad(i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class DWSepConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', stride=1, dilation=1, bias=False):
        super().__init__()
        if padding == 'same':
            self.depthwise = Conv2DSamePad(in_channels, in_channels, kernel_size=kernel_size,
                                           stride=stride, dilation=dilation, groups=in_channels, bias=bias)
        else:
            self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                       stride=stride, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.relu(out)
        return out


class BaseConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', stride=1, dilation=1, bias=False):
        super().__init__()
        if padding == 'same':
            self.conv = Conv2DSamePad(in_channels, out_channels, kernel_size=kernel_size,
                                      stride=stride, dilation=dilation, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out


class Model(nn.Module):

    def __init__(self, in_channels, num_classes, embedding_dims, num_filters):
        super().__init__()

        # Downscale module
        self.ds_conv1 = DWSepConv2D(in_channels, num_filters, stride=2, dilation=1)
        self.ds_conv2 = DWSepConv2D(num_filters, num_filters, stride=1, dilation=1)
        self.ds_conv3 = DWSepConv2D(num_filters, num_filters, stride=2, dilation=1)

        # Context module
        self.ct_conv1 = BaseConv2D(num_filters, num_filters, stride=1, dilation=1)
        self.ct_conv2 = BaseConv2D(num_filters, num_filters, stride=1, dilation=2)
        self.ct_conv3 = BaseConv2D(num_filters, num_filters, stride=1, dilation=4)
        self.ct_conv4 = BaseConv2D(num_filters, num_filters, stride=1, dilation=8)
        self.ct_conv4 = BaseConv2D(num_filters, num_filters, stride=1, dilation=16)
        self.ct_conv5 = BaseConv2D(num_filters, num_filters, stride=1, dilation=1)

        # Final
        self.base_conv = nn.Conv2d(num_filters, num_classes + 1, kernel_size=1)
        self.embed_conv = nn.Conv2d(num_filters, embedding_dims, kernel_size=1)

    def forward(self, x):
        x = self.ds_conv1(x)
        x = self.ds_conv2(x)
        x = self.ds_conv3(x)

        x = self.ct_conv1(x)
        x = self.ct_conv2(x)
        x = self.ct_conv3(x)
        x = self.ct_conv4(x)
        x = self.ct_conv5(x)

        base_out = self.base_conv(x)
        embeddings_out = self.embed_conv(x)

        return base_out, embeddings_out

