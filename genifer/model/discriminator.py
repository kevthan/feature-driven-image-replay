"""
Discriminator model that distinguished real from fake samples.
"""

import torch.nn as nn


class BaseCondDiscriminator(nn.Module):

    def __init__(
        self, img_input_channels, img_input_size, device, n_cls=0, prev_model=None
    ):
        super(BaseCondDiscriminator, self).__init__()
        self.device = device

    def forward(self, x, c):
        raise NotImplementedError("Implemented by subclasses.")

    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def _init_from_prev_model(self, prev_model):
        prev_state_dict = prev_model.state_dict()
        model_dict = self.model.state_dict()

        for param_name in model_dict.keys():
            model_dict[param_name] = prev_state_dict[param_name]

        self.model.load_state_dict(model_dict)
