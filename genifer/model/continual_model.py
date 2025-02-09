import torch
import torch.nn as nn
import importlib

from genifer.utils.io import load_feature_extractor_checkpoint
from genifer.dataloader.constants import INPUT_DIMENSIONS


# Available trainers
AVAILABLE_MODELS = {
    "resnet": ["backbones", "ResNet"],
}


class ContinualModel(nn.Module):

    def __init__(self, config, n_classes, device, prev_model=None):
        super(ContinualModel, self).__init__()

        model_name = config["model_params"]["name"]
        n_layers = config["model_params"]["n_layers"]
        activation = config["model_params"].get("activation", "relu")
        normalized = config["model_params"].get("normalized", True)
        pretrained = config["model_params"].get("pretrained", False)
        last_layer_to_load = config["model_params"].get("last_layer_to_load", None)
        num_features = config["model_params"]["latent_dim"]
        bias = config["model_params"]["bias"]
        input_dim = INPUT_DIMENSIONS[config["data_params"]["dataset_name"]]

        # build feature extractor
        feat_extractor_module = importlib.import_module(
            "genifer.model.{}".format(AVAILABLE_MODELS[model_name][0])
        )
        feat_extractor_class = getattr(
            feat_extractor_module, AVAILABLE_MODELS[model_name][1]
        )
        self.feature_extractor = feat_extractor_class(
            input_dim=input_dim,
            latent_dim=num_features,
            activation=activation,
            normalized=normalized,
            pretrained=pretrained,
            n_layers=n_layers,
            device=device,
        )

        # load feature extractor if path is provided
        if pretrained and isinstance(pretrained, str) and prev_model is None:
            load_feature_extractor_checkpoint(self, pretrained, last_layer_to_load)

        self.n_prev_cls = 0 if prev_model is None else prev_model.n_cls
        self.n_cls = n_classes

        self.prev_readout = nn.Linear(
            in_features=num_features, out_features=self.n_prev_cls, bias=bias
        )
        self.curr_readout = nn.Linear(
            in_features=num_features,
            out_features=self.n_cls - self.n_prev_cls,
            bias=bias,
        )
        self.readout_param_types = ["weight", "bias"] if bias else ["weight"]
        self.reinit_starting_layer = config["model_params"].get(
            "reinit_starting_layer", None
        )

        if prev_model is not None:
            self._init_from_prev_model(prev_model)

        self.device = device
        self.to(device)

    def forward(self, x, input_stage="stage0", output_stage=None):
        if output_stage is None:
            h = self.feature_extractor(x, input_stage)
            output = torch.cat((self.prev_readout(h), self.curr_readout(h)), dim=1)
            return output, h
        else:
            h, feat = self.feature_extractor(x, output_stage=output_stage)
            output = torch.cat((self.prev_readout(h), self.curr_readout(h)), dim=1)
            return output, h, feat

    def get_curr_logits(self, x):
        logits, h = self.forward(x)
        return logits[:, self.n_prev_cls :], h

    def get_prev_logits(self, x):
        logits, h = self.forward(x)
        return logits[:, : self.n_prev_cls], h

    def freeze_prev_output_params(self):
        self._set_prev_output_params(False)

    def unfreeze_prev_output_params(self):
        self._set_prev_output_params(True)

    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def _set_prev_output_params(self, value):
        for param in self.prev_readout.parameters():
            param.requires_grad = value

    def _init_from_prev_model(self, prev_model):
        """
        Initialize from a previous model.

        :param prev_model:  Previous model
        :return:            None
        """
        prev_state_dict = prev_model.state_dict()
        model_dict = self.state_dict()

        # Get state dict entries of the previous feature extractor
        for param_name in model_dict.keys():
            if (
                self.reinit_starting_layer is not None
                and self.reinit_starting_layer in param_name
            ):
                return self.load_state_dict(model_dict, strict=False)
            if "feature_extractor" in param_name:
                model_dict[param_name] = prev_state_dict[param_name]

        # Get readout weights and biases from previous model
        for param_type in self.readout_param_types:
            prev_param_name = "prev_readout.{}".format(param_type)
            curr_param_name = "curr_readout.{}".format(param_type)
            all_prev_params = torch.cat(
                (prev_state_dict[prev_param_name], prev_state_dict[curr_param_name])
            )
            model_dict[prev_param_name] = all_prev_params

        return self.load_state_dict(model_dict, strict=False)

    @staticmethod
    def get_activations(name, activations):

        def hook(module, input, output):
            activations[name] = output

        return hook
