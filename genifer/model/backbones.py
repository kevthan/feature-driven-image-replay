from copy import deepcopy

import torch.nn as nn
import torchvision


class BackBones(nn.Module):

    def __init__(
        self,
        input_dim,
        n_layers,
        latent_dim=2,
        activation="relu",
        normalized=False,
        pretrained=False,
        device="cpu",
    ):
        super(BackBones, self).__init__()
        self.to(device)

    def forward(self, x):
        raise NotImplementedError("Implementation in subclasses.")


class ResNet(BackBones):

    def __init__(
        self,
        input_dim,
        n_layers,
        latent_dim=2,
        activation="relu",
        normalized=False,
        pretrained=False,
        device="cpu",
    ):

        super(ResNet, self).__init__(
            input_dim, n_layers, latent_dim, activation, normalized, pretrained, device
        )

        self.img_ch = input_dim[0]
        self.img_size = input_dim[1]

        if n_layers == 18:
            model = torchvision.models.resnet18(pretrained=pretrained)
            channels = [64, 64, 128, 256, 512]
        elif n_layers == 34:
            model = torchvision.models.resnet34(pretrained=pretrained)
            channels = [64, 64, 128, 256, 512]
        elif n_layers == 50:
            model = torchvision.models.resnet50(pretrained=pretrained)
            channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(
                "ResNet with {} layers is not implemented!".format(n_layers)
            )

        # Consolidate layers to stages
        self.stage0 = nn.Identity()
        if self.img_size == 32:
            conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.stage1 = nn.Sequential(*[conv1, deepcopy(model.bn1), nn.ReLU()])
            self.stage2 = nn.Sequential(*[deepcopy(model.layer1)])
            self.stage_info = {
                "stage0": [self.img_ch, self.img_size],
                "stage1": [channels[0], self.img_size],
                "stage2": [channels[1], self.img_size],
                "stage3": [channels[2], self.img_size // 2],
                "stage4": [channels[3], self.img_size // 4],
                "stage5": [channels[4], self.img_size // 8],
                "stage6": [latent_dim, 1],
            }
        else:
            self.stage1 = nn.Sequential(
                *[deepcopy(model.conv1), deepcopy(model.bn1), nn.ReLU()]
            )
            self.stage2 = nn.Sequential(
                *[
                    deepcopy(model.layer1),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                ]
            )
            self.stage_info = {
                "stage0": [self.img_ch, self.img_size],
                "stage1": [channels[0], self.img_size // 2],
                "stage2": [channels[1], self.img_size // 4],
                "stage3": [channels[2], self.img_size // 8],
                "stage4": [channels[3], self.img_size // 16],
                "stage5": [channels[4], self.img_size // 32],
                "stage6": [latent_dim, 1],
            }
        self.stage3 = deepcopy(model.layer2)
        self.stage4 = deepcopy(model.layer3)
        self.stage5 = deepcopy(model.layer4)
        self.stage6 = nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()])

    def forward(self, x, input_stage="stage0", output_stage=None):

        reached_input_stage = False
        out_feat = None
        for name, module in self.named_children():
            if reached_input_stage:
                x = module(x)
            if name == input_stage:
                reached_input_stage = True
            if name == output_stage:
                out_feat = x
        if output_stage is None:
            return x
        else:
            return x, out_feat
