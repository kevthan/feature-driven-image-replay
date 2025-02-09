"""
Base class for different continual learning trainers.
"""

import torch


class BaseTrainer:

    def __init__(self, model, config, train_loader, model_path, ml_logger=None):
        self.model = model
        self.model_path = model_path
        self.train_loader = train_loader
        self.task_id = 0
        self.ml_logger = ml_logger

    def update(self, model, train_loader):
        """
        Updates parameters and the model in the trainer.

        :param model:           Model
        :param train_loader:    Training set dataloader
        :return:                None
        """
        self.model = model
        self.train_loader = train_loader
        self.task_id += 1

    def get_modified_loss(self, samples, labels, curr_loss_fn, step=None, i_batch=None):
        """
        Construct a possibly modified loss.

        :param samples:         Batch of samples
        :param labels:          Batch of labels corresponding to the samples
        :param curr_loss_fn:    Loss function for current samples
        :return:                The modified loss
        """
        logits = self.model(samples)[0]
        curr_loss = curr_loss_fn(logits, labels)
        cl_loss = None
        return curr_loss, cl_loss

    def batch_finished(self, loss):
        """
        Execute operations after training on a batch.

        :param loss:    Original loss
        :return:        None
        """
        pass

    def prepare_next_training(self, data_loader):
        """
        Execute operations after training on a task.

        :param data_loader:    Data loader
        """
        pass

    def test(self):
        """
        Execute operations at test time.
        :return:
        """
        pass

    def _zero_pad_readout(self, dict1, dict2):
        """
        Zero-pad readout of dict1 to match dimensions of dict2.

        :param dict1:   Dictionary with readout that needs to be zero-padded (previous model)
        :param dict2:   Reference dictionary (current model)
        :return:        Zero-padded dict1
        """
        for param_type in self.model.readout_param_types:
            param_name = "readout.{}".format(param_type)
            dict1_readout_shape = dict1[param_name].shape
            dict2_readout_shape = dict2[param_name].shape
            if param_type == "weight":
                padding = torch.zeros(
                    dict2_readout_shape[0] - dict1_readout_shape[0],
                    dict2_readout_shape[1],
                ).type(torch.FloatTensor)
            else:
                padding = torch.zeros(
                    dict2_readout_shape[0] - dict1_readout_shape[0]
                ).type(torch.FloatTensor)
            if dict1[param_name].is_cuda and dict2[param_name].is_cuda:
                padding = padding.cuda()
            dict1[param_name] = torch.cat((dict1[param_name], padding))
        return dict1
