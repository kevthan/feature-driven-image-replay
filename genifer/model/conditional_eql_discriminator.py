import torch
from torch import nn as nn
import numpy as np

from .discriminator import BaseCondDiscriminator
from .layers import EQConv2d, EQLinear


class Prologue(nn.Module):
    def __init__(self, stage, out_nchan=512):
        super().__init__()
        # stage in range(0, 6)
        self.stage = stage
        # cheap downsampling
        self.biliDown = nn.AvgPool2d(2, stride=2)
        # Activation function
        self.act = nn.LeakyReLU(0.2)

        if stage in (0, 2):
            in_n_chan = 3 if stage == 0 else 128
            base = 128
            # in_n_chanx32x32 -> 128x32x32
            self.conv1 = EQConv2d(
                in_n_chan, base, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 128x32x32 -> 128x32x32
            self.conv2 = EQConv2d(
                base, base, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 128x16x16 -> 256x16x16
            self.conv3 = EQConv2d(
                base, base * 2, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 256x16x16 -> 256x16x16
            self.conv4 = EQConv2d(
                base * 2, base * 2, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 256x8x8 -> 512x8x8
            self.conv5 = EQConv2d(
                base * 2, base * 4, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 512x8x8 -> 512x8x8
            self.conv6 = EQConv2d(
                base * 4, base * 4, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 512x4x4 -> 512x4x4
            self.conv7 = EQConv2d(
                base * 4, out_nchan, kernel_size=3, stride=1, padding=1, bias=True
            )

        if stage == 3:
            in_n_chan = 128
            base = 128
            # 128x16x16 -> 128x16x16
            self.conv1 = EQConv2d(
                in_n_chan, base, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 128x8x8 -> 256x8x8
            self.conv2 = EQConv2d(
                base, base * 2, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 256x8x8 -> 256x8x8
            self.conv3 = EQConv2d(
                base * 2, base * 2, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 256x4x4 -> 512x4x4
            self.conv4 = EQConv2d(
                base * 2, base * 4, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 512x4x4 -> 512x4x4
            self.conv5 = EQConv2d(
                base * 4, base * 4, kernel_size=3, stride=1, padding=1, bias=True
            )

        if stage == 4:
            in_n_chan = 256
            base = 256
            # 256x8x8 -> 256x8x8
            self.conv1 = EQConv2d(
                in_n_chan, base, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 256x8x8 -> 256x8x8
            self.conv2 = EQConv2d(
                base, base, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 256x4x4 -> 512x4x4
            self.conv3 = EQConv2d(
                base, base * 2, kernel_size=3, stride=1, padding=1, bias=True
            )
            # 512x4x4 -> 512x4x4
            self.conv4 = EQConv2d(
                base * 2, out_nchan, kernel_size=3, stride=1, padding=1, bias=True
            )

        if stage == 5:
            in_n_chan = 512
            base = 1024
            self.fc1 = EQLinear(in_n_chan, base, bias=True)
            self.fc2 = EQLinear(base, base, bias=True)
            self.fc3 = EQLinear(base, base, bias=True)
            self.fc4 = EQLinear(base, base, bias=True)
            self.fc5 = EQLinear(base, base, bias=True)
            self.fc6 = EQLinear(base, base, bias=True)
            self.fc7 = EQLinear(base, base, bias=True)
            self.fc8 = EQLinear(base, base, bias=True)
            self.fc9 = EQLinear(base, out_nchan, bias=True)

    def forward(self, x):
        # input matching
        if self.stage in (0, 2):
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = self.biliDown(x)
            x = self.act(self.conv3(x))
            x = self.act(self.conv4(x))
            x = self.biliDown(x)
            x = self.act(self.conv5(x))
            x = self.act(self.conv6(x))
            x = self.biliDown(x)
            x = self.act(self.conv7(x))
            x = self.biliDown(x)
            return x

        if self.stage == 3:
            # 128x16x16 -> 128x16x16
            x = self.act(self.conv1(x))
            # 128x16x16 -> 128x8x8
            x = self.biliDown(x)
            # 128x8x8 -> 256x8x8
            x = self.act(self.conv2(x))
            x = self.act(self.conv3(x))
            # 256x8x8 -> 256x4x4
            x = self.biliDown(x)
            # 256x4x4 -> 512x4x4
            x = self.act(self.conv4(x))
            x = self.act(self.conv5(x))
            x = self.biliDown(x)
            return x

        if self.stage == 4:
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = self.biliDown(x)
            x = self.act(self.conv3(x))
            x = self.act(self.conv4(x))
            x = self.biliDown(x)
            return x

        if self.stage == 5:
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            x = self.act(self.fc4(x))
            x = self.act(self.fc5(x))
            x = self.act(self.fc6(x))
            x = self.act(self.fc7(x))
            x = self.act(self.fc8(x))
            x = self.act(self.fc9(x))
            return x


class Epilogue(nn.Module):
    def __init__(
        self,
        stage,
        in_channels,
        res,
        n_cls=0,
        mbdis_group_size=32,
        mbdis_n_chan=0,
        cmap_dim=128,
    ):
        super().__init__()
        self.cmap_dim = cmap_dim
        self.stage = stage
        self.mbdis = None
        self.act = nn.LeakyReLU(0.2)

        # handcrafted mbdis features
        self.mbdis = (
            MinibatchStdLayer(mbdis_group_size, mbdis_n_chan)
            if mbdis_n_chan > 0
            else None
        )
        if stage == 5:
            self.dense1 = EQLinear(in_channels + mbdis_n_chan, in_channels, bias=True)
            self.dense2 = EQLinear(in_channels, in_channels // 2, bias=True)
            # output layer (maps to cmap_dim outputs instead of single logit)
            self.logits = EQLinear(in_channels // 2, cmap_dim, bias=True)
        else:
            # last conv layer incorporates mbdis features
            self.conv = EQConv2d(
                in_channels + mbdis_n_chan, in_channels, kernel_size=3, padding=1
            )
            # dense layer instead of further downsampling
            self.dense = EQLinear(in_channels * (res**2), in_channels, bias=True)
            # output layer (maps to cmap_dim outputs instead of single logit)
            self.logits = EQLinear(in_channels, cmap_dim, bias=True)

        # projection layer for condition label (affine)
        self.onehot_project = EQLinear(n_cls, cmap_dim, bias=False)

    def forward(self, x, c):
        if self.mbdis is not None:
            if self.stage == 5:
                C = x.size()[1]
                x = torch.reshape(x, (-1, C, 1, 1))
                x = self.mbdis(x)
                x = x.flatten(1)
            else:
                x = self.mbdis(x)

        if self.stage == 5:
            x = self.act(self.dense1(x))
            x = self.act(self.dense2(x))
        else:
            x = self.act(self.conv(x))
            # dense layer that does 'downsampling'
            x = self.act(self.dense(x.flatten(1)))

        logits = self.logits(x)

        # project condition
        c_proj = self.onehot_project(c)
        out = (logits * c_proj).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


class MinibatchStdLayer(nn.Module):
    def __init__(self, group_size, n_chan=1):
        super().__init__()
        self.group_size = group_size
        self.n_chan = n_chan

    def forward(self, x):
        N, C, H, W = x.shape
        G = N
        if self.group_size is not None:
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N))
        F = self.n_chan
        c = C // F

        # split minibatch in n groups of size G, split channels in F groups of size c
        y = x.reshape(G, -1, F, c, H, W)
        # shift center (per group) to zero
        y = y - y.mean(dim=0)
        # variance per group
        y = y.square().mean(dim=0)
        # stddev
        y = (y + 1e-8).sqrt()
        # average over channels and pixels
        y = y.mean(dim=[2, 3, 4])
        # reshape and tile
        y = y.reshape(-1, F, 1, 1)
        y = y.repeat(G, 1, H, W)
        # add to input as 'handcrafted feature channels'
        x = torch.cat([x, y], dim=1)
        return x


class CondEqlDiscriminator(BaseCondDiscriminator):

    def __init__(
        self,
        img_input_channels,
        img_input_size,
        device,
        n_cls=0,
        prev_model=None,
        mbdis=True,
    ):
        super(CondEqlDiscriminator, self).__init__(
            img_input_channels=img_input_channels,
            img_input_size=img_input_size,
            n_cls=n_cls,
            device=device,
            prev_model=prev_model,
        )

        # input dimensions
        self.in_size = img_input_size
        self.in_channels = img_input_channels
        n_cls = n_cls
        # prologue, depending on matching stage
        stage = 0
        stage = 2 if self.in_size == 32 and self.in_channels == 64 else stage
        stage = 3 if self.in_size == 16 and self.in_channels == 128 else stage
        stage = 4 if self.in_size == 8 and self.in_channels == 256 else stage
        stage = 5 if self.in_size == 4 and self.in_channels == 512 else stage
        print("matching stage{}".format(stage))
        # interface between prologue and epilogue
        epilogue_in_res = 2
        epilogue_in_nchan = 1024 if stage == 5 else 512
        self.prologue = Prologue(stage, out_nchan=epilogue_in_nchan)
        # epilogue, with optional mbdis
        mbdis_n_chan = 1 if mbdis else 0
        self.logits = Epilogue(
            stage,
            epilogue_in_nchan,
            n_cls=n_cls,
            res=epilogue_in_res,
            mbdis_n_chan=mbdis_n_chan,
        )

        self.to(device)

    def accepts(self, data):

        assert len(data.shape) == 4

        # Return true if the discriminator can process the given data as input
        return list(data.shape[1:-1]) == [self.in_channels, self.in_size]

    def forward(self, x, c):
        x = self.prologue(x)
        # epilogue
        x = self.logits(x, c)

        return x
