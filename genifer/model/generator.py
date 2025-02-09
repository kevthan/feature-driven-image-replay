"""
Generative model that produces artificial samples and labels.
"""

import torch.nn as nn


class BaseGenerator(nn.Module):

    def __init__(
        self,
        latent_dim,
        image_channels,
        image_size,
        device,
        n_cls=0,
        matching_mode="indirect",
        prev_model=None,
    ):
        super(BaseGenerator, self).__init__()
        self.device = device

    def forward(self, x):
        raise NotImplementedError("Implemented by subclasses.")

    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def _init_from_prev_model(self, prev_model):
        prev_state_dict = prev_model.state_dict()
        model_dict = self.state_dict()

        for param_name in model_dict.keys():
            model_dict[param_name] = prev_state_dict[param_name]

        self.load_state_dict(model_dict)
