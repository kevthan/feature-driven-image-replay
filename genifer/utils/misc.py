import numpy as np


def get_n_trainable_params(model, last_stage=None):
    if last_stage is not None:
        named_params = filter(
            lambda p: p[1].requires_grad
            and "stage" in p[0]
            and int(p[0].split("stage")[1][0]) <= last_stage,
            model.named_parameters(),
        )
        return sum([np.prod(p.size()) for n, p in named_params])
    else:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])


def flatten_config(config):
    """
    Flatten a configuration dictionary.

    :param config:  Configuration dictionary
    :return:        Flat configuration dictionary
    """
    flat_config = {}
    for param_type, param_dict in config.items():
        for sub_param_name, sub_param_value in param_dict.items():
            if isinstance(sub_param_value, dict):
                for subsub_param_name, subsub_param_value in sub_param_value.items():
                    flat_config[
                        "{}.{}.{}".format(param_type, sub_param_name, subsub_param_name)
                    ] = subsub_param_value
            else:
                flat_config["{}.{}".format(param_type, sub_param_name)] = (
                    sub_param_value
                )
    return flat_config
