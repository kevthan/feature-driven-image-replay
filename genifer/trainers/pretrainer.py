"""
Pretrainer.
"""

import sys
import logging

from genifer.trainers.base_trainer import BaseTrainer
from genifer.utils.io import get_device

log = logging.getLogger("CL::Pretrainer")
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
h1.setFormatter(formatter)
log.addHandler(h1)


class Pretrainer(BaseTrainer):

    def __init__(self, model, model_path, config, train_loader, ml_logger):
        super(Pretrainer, self).__init__(
            model=model,
            model_path=model_path,
            config=config,
            train_loader=train_loader,
            ml_logger=ml_logger,
        )
        self.freeze_layers = config["cl_params"]["method"].get("freeze_layers", False)
        self.batch_size = config["data_params"]["batch_size"]
        self.seed = config["misc_params"]["seed"]
        self.prev_model = None
        self.device = get_device()

    def get_modified_loss(self, samples, labels, curr_loss_fn, step=None, i_batch=None):
        # no continual learning loss for pretraining
        logits = self.model(samples)[0]
        curr_loss = curr_loss_fn(logits, labels)
        cl_loss = None
        return curr_loss, cl_loss
