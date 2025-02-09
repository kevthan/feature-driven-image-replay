import torch
from torch import nn

from .generator import BaseGenerator
from .layers import StyleEQConv2dWithBias, EQLinear


def scale_to_hypersphere(x, dim=1, eps=1e-8):
    return x


class CondEqlGenerator(BaseGenerator):

    def __init__(
        self,
        latent_dim,
        image_channels,
        image_size,
        device,
        n_cls=0,
        use_task_labels=False,
        matching_mode="indirect",
        prev_model=None,
    ):
        super(CondEqlGenerator, self).__init__(
            latent_dim=latent_dim,
            image_channels=image_channels,
            image_size=image_size,
            device=device,
            n_cls=n_cls,
            matching_mode=matching_mode,
            prev_model=prev_model,
        )

        self.latent_dim = latent_dim
        # dimension of linearly embedded onehot
        onehot_embed_d = 256
        # activation used whenever non-linear
        self.act = nn.LeakyReLU(0.2)
        # bilinear upsampling layer
        self.biliUp = nn.UpsamplingBilinear2d(scale_factor=2)

        # embed onehot
        self.onehot_embed = EQLinear(n_cls, onehot_embed_d, bias=True)
        # map latent + onehot to w
        self.w1 = EQLinear(
            latent_dim + onehot_embed_d, latent_dim, bias=True, lr_mul=0.01
        )
        self.w2 = EQLinear(latent_dim, latent_dim, bias=True, lr_mul=0.01)

        # no. of in channels foreach layer
        in_nchans = [None, 512, 512, 256, 256, 128, 128, 64, 64]
        base_res = 4
        base_nchan = 512
        # learned constant
        self.const = nn.Parameter(torch.ones((1, base_nchan, base_res, base_res)))
        # conv layers, styles, and noise scales
        for i in range(1, 8):
            in_nchan = in_nchans[i]
            out_nchan = in_nchans[i + 1]
            conv = StyleEQConv2dWithBias(
                in_nchan,
                out_nchan,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                device=device,
            )
            setattr(self, "conv{}".format(i), conv)
        # output layer (no noise)
        self.out_layer = StyleEQConv2dWithBias(
            in_nchans[-1], 3, kernel_size=3, stride=1, padding=1, bias=True, noise=False
        )

        self.to(device)

    def forward(self, x, noise=None):
        # split in z, onehot
        z = x[:, : self.latent_dim]
        one_hot = x[:, self.latent_dim :]

        # embed (linearly), normalize, and concat
        one_hot = self.onehot_embed(one_hot)
        one_hot = scale_to_hypersphere(one_hot)
        z = scale_to_hypersphere(z)
        x = torch.cat([z, one_hot], dim=1)

        # map to w
        w = self.act(self.w1(x))
        w = self.act(self.w2(w))

        # conv. part.
        bs = x.size()[0]
        # broadcast learned constant along batch dim
        x = self.const.expand([bs, -1, -1, -1])
        for i in range(1, 8):
            style_conv = getattr(self, "conv{}".format(i))
            n = noise[i] if noise is not None else None
            x = style_conv((x, w), n=n)
            x = self.act(x)
            if i % 2 == 0:
                x = self.biliUp(x)

        # linear output
        return self.out_layer((x, w))
