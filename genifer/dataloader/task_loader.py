"""Classes that enable data loading for different tasks."""

import importlib
import torch
import torch.utils.data as data

from genifer.dataloader.constants import DATASET_CLASSES
from genifer.utils.io import get_worker_init_fn


class TaskLoader:

    def __init__(
        self,
        split_info,
        task_id,
        dataset_name,
        dataset_path,
        normalization,
        device,
        batch_size,
        seed,
    ):
        self.task_id = task_id
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.DatasetClass = self._get_dataset_class()
        self.normalization = normalization
        self.device = device
        self.split_info = split_info
        self.batch_size = batch_size
        self.n_prev_curr_cls = self._get_n_cls()
        self.seed = seed
        torch.manual_seed(self.seed)

    def get_data_loaders(self, shuffle=True):
        """
        Get patch data loaders (train, test & val) for this task.

        :param shuffle:             Whether to shuffle the data
        :return:                    Dictionary of patch data loaders
        """

        train_sampler = None

        # Only use train data of the current task
        print("Load training set...")
        train_set = self.DatasetClass(
            class_split=self.split_info,
            task_id=self.task_id,
            without_offset=False,
            split="train",
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            normalization=self.normalization,
            device=self.device,
            seed=self.seed,
        )

        # For reproducibility, get init function for data loading worker(s)
        worker_init_fn = get_worker_init_fn(self.seed)

        # Create data loader for training
        train_loader = data.DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            shuffle=shuffle if train_sampler is None else False,
            sampler=train_sampler,
            num_workers=8,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )

        # Test data loaders contain all test data from current and previous tasks
        test_loader_list = []
        for t_id in range(self.task_id + 1):
            print("Load test set of task {}...".format(t_id))
            test_set = self.DatasetClass(
                class_split=self.split_info,
                task_id=t_id,
                without_offset=False,
                split="test",
                dataset_name=self.dataset_name,
                dataset_path=self.dataset_path,
                normalization=self.normalization,
                device=self.device,
                seed=self.seed,
            )
            test_loader = data.DataLoader(
                dataset=test_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                worker_init_fn=worker_init_fn,
            )

            test_loader_list.append(test_loader)

        return {"train": train_loader, "test": test_loader_list}

    def _get_n_cls(self):
        n_prev_curr_cls = 0
        for key, val in self.split_info.items():
            if key <= self.task_id:
                n_prev_curr_cls += len(val)
        return n_prev_curr_cls

    def _get_dataset_class(self):
        """
        Import the dataset class corresponding to the given dataset.

        :return:    Dataset class
        """
        dataset_module = importlib.import_module(
            DATASET_CLASSES[self.dataset_name]["module"]
        )
        dataset_class = getattr(
            dataset_module, DATASET_CLASSES[self.dataset_name]["class"]
        )
        return dataset_class
