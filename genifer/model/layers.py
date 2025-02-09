# adapted from https://github.com/akanimax/pro_gan_pytorch

import torch
import numpy as np
from typing import Any
from torch.nn import Conv2d, ConvTranspose2d, Linear
from torch import Tensor
from torch import nn


class EQConv2d(Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        # initialize weights from Normal
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # eq lr
        fan_in = np.prod(self.kernel_size) * self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor) -> Tensor:
        return torch.conv2d(
            input=x,
            weight=self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class StyleEQConv2d(EQConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        wdim=512,
        stylemod=True,
        noise=True,
        device=None,
    ) -> None:
        super(StyleEQConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.device = device
        self.out_channels = out_channels
        # No bias, scale only
        self.y = EQLinear(wdim, in_channels, bias_init=1.0) if stylemod else None
        # Single noise scalar
        self.noise_scale = (
            nn.Parameter(torch.zeros([1, out_channels, 1, 1])) if noise else None
        )

    def forward(self, x: tuple, n=None) -> Tensor:
        x, w = x
        bs, nchan, res = x.size()[:3]
        # Style modulation
        if self.y is not None:
            y = self.y(w)
            y = y.reshape(-1, nchan, 1, 1)
            x = x * y
        # Convolution
        x = super(StyleEQConv2d, self).forward(x)
        # Add noise
        if self.noise_scale is not None:
            n = torch.randn((bs, 1, res, res), device=self.device) if n is None else n
            x += self.noise_scale * n

        return x


class StyleEQConv2dWithBias(EQConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        wdim=512,
        stylemod=True,
        noise=True,
        device=None,
    ) -> None:
        super(StyleEQConv2dWithBias, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.device = device
        self.out_channels = out_channels
        # No bias, scale only
        self.y1 = EQLinear(wdim, in_channels, bias_init=1.0) if stylemod else None
        # Bias
        self.y2 = EQLinear(wdim, in_channels, bias_init=0.0) if stylemod else None
        # Single noise scalar
        self.noise_scale = (
            nn.Parameter(torch.zeros([1, out_channels, 1, 1])) if noise else None
        )

    def forward(self, x: tuple, n=None) -> Tensor:
        x, w = x
        bs, nchan, res = x.size()[:3]
        # Style modulation
        if self.y1 is not None and self.y2 is not None:
            y1 = self.y1(w)
            y2 = self.y2(w)
            y1 = y1.reshape(-1, nchan, 1, 1)
            y2 = y2.reshape(-1, nchan, 1, 1)
            x = x * y1 + y2

        # Convolution
        x = super(StyleEQConv2dWithBias, self).forward(x)
        # Add noise
        if self.noise_scale is not None:
            n = torch.randn((bs, 1, res, res), device=self.device) if n is None else n
            x += self.noise_scale * n

        return x


class EQConvTranspose2d(ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )

        # init weights from Normal
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # eq lr
        fan_in = self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor, output_size: Any = None) -> Tensor:
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size
        )
        return torch.conv_transpose2d(
            input=x,
            weight=self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )


class EQLinear(Linear):
    def __init__(
        self, in_features, out_features, bias=True, bias_init=0.0, lr_mul=1.0
    ) -> None:
        super().__init__(in_features, out_features, bias)

        # init weights from Normal
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0 / lr_mul)

        # init bias
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )

        # eq lr
        fan_in = self.in_features
        self.weight_scale = lr_mul / np.sqrt(fan_in)
        self.bias_scale = lr_mul

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight * self.weight_scale
        if self.bias is not None and self.bias_scale is not 1:
            b = self.bias * self.bias_scale
        else:
            b = self.bias
        return torch.nn.functional.linear(x, w, b)
