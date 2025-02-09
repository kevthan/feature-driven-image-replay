"""Classes that enable data loading of the torchvision datasets."""

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.datasets as datasets
from .constants import (
    NORMALIZATION_FACTORS,
    DEFAULT_NORMALIZATION_FACTORS,
    TRAIN_FRACTION,
)


class TorchVisionDataset(data.Dataset):

    def __init__(
        self,
        class_split,
        task_id,
        indices_per_label=None,
        without_offset=True,
        split="train",
        dataset_name="MNIST",
        normalization="default",
        device="cpu",
        dataset_path="./data",
        seed=2424,
    ):

        # 1. set class attributes
        if normalization == "known":
            print("Using known normalization!")
            if dataset_name in NORMALIZATION_FACTORS.keys():
                self.mean = NORMALIZATION_FACTORS[dataset_name]["mean"]
                self.std = NORMALIZATION_FACTORS[dataset_name]["std"]
            else:
                raise NotImplementedError(
                    "Dataset: {} does not have known normalization factors. Please provide them.".format(
                        dataset_name
                    )
                )
        elif normalization == "default":
            if dataset_name in DEFAULT_NORMALIZATION_FACTORS.keys():
                self.mean = DEFAULT_NORMALIZATION_FACTORS[dataset_name]["mean"]
                self.std = DEFAULT_NORMALIZATION_FACTORS[dataset_name]["std"]
            else:
                raise NotImplementedError(
                    "Dataset: {} does not have default normalization factors. Please provide them.".format(
                        dataset_name
                    )
                )
        else:
            raise ValueError("Invalid normalization option '{}'".format(normalization))

        if "MNIST" in dataset_name:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
            )
        else:
            if split == "train":
                self.transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4, padding_mode="edge"),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(self.mean, self.std),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
                )

        self.class_split = class_split
        self.without_offset = without_offset
        self.task_id = task_id
        self.device = device
        self.seed = seed

        # 2. load data
        data = getattr(datasets, dataset_name)(
            root=dataset_path, train=split == "train" or split == "val", download=True
        )

        # 3. Get data for the current task
        if indices_per_label is not None:
            # This dataset is an exemplar set
            assert split == "train" or split == "val"

            self.images, self.labels = [], []
            classes = []
            for prev_task_id in range(self.task_id):
                # Regenerate images and labels of a previous task
                prev_images, prev_labels = self.extract_subdata(
                    all_images=[x[0] for x in data],
                    all_labels=[x[1] for x in data],
                    task_id=prev_task_id,
                    split=split,
                )

                # Select only the images and labels defined by the exemplar sampler
                for label in self.class_split[prev_task_id].values():
                    self.images += [prev_images[i] for i in indices_per_label[label]]
                    self.labels += [prev_labels[i] for i in indices_per_label[label]]
                    classes.append(label)
            self.classes = np.array(classes)

        else:
            # This dataset is NOT an exemplar set
            self.images, self.labels = self.extract_subdata(
                all_images=[x[0] for x in data],
                all_labels=[x[1] for x in data],
                task_id=self.task_id,
                split=split,
            )
            self.classes = np.array(list(self.class_split[task_id].keys()))

        self.class_to_idx = {cls: i_cls for (i_cls, cls) in enumerate(self.classes)}
        self.idx_to_class = {i_cls: cls for (i_cls, cls) in enumerate(self.classes)}
        self.mapped_class_to_idx = {
            cls: i_cls for (i_cls, cls) in enumerate(self.class_split[task_id].values())
        }
        self.idx_to_mapped_class = {
            i_cls: cls for (i_cls, cls) in enumerate(self.class_split[task_id].values())
        }

        print("class_to_idx", self.class_to_idx)
        print("mapped_class_to_idx", self.mapped_class_to_idx)

    def __getitem__(self, index):
        img = self.transform(self.images[index])
        cls = self.labels[index]
        if self.without_offset:
            cls = self.mapped_class_to_idx[cls]
        return img, cls

    def __len__(self):
        return len(self.labels)

    def get_labels_without_offset(self, labels):
        """
        Map the labels that are assigned relative to all tasks
        to new labels starting from 0, e.g.:

        (k, k+1, ..., k+m) --> (0, 1, ..., m)

        :param labels:  Tensor of given labels (w.r.t. all tasks)
        :return:        Tensor of new labels (w.r.t. current task)
        """
        new_labels = torch.LongTensor(
            [self.mapped_class_to_idx[l.item()] for l in labels]
        ).to(self.device)
        return new_labels

    def extract_subdata(self, all_images, all_labels, task_id, split):
        """
        Extract the data with labels subclass from
        the list of images and labels.
        """
        if split == "train" or split == "val":
            labels, images = [], []
            for label in self.class_split[task_id].keys():
                train_val_labels = [
                    self.class_split[task_id][lab] for lab in all_labels if lab == label
                ]
                train_val_images = [
                    image
                    for (i, image) in enumerate(all_images)
                    if all_labels[i] == label
                ]

                n_train = int(TRAIN_FRACTION * len(train_val_labels))
                if split == "train":
                    labels += train_val_labels[:n_train]
                    images += train_val_images[:n_train]
                else:
                    labels += train_val_labels[n_train:]
                    images += train_val_images[n_train:]

        elif split == "test":
            labels = [
                self.class_split[task_id][label]
                for label in all_labels
                if label in self.class_split[task_id].keys()
            ]
            images = [
                image
                for (i, image) in enumerate(all_images)
                if all_labels[i] in self.class_split[task_id].keys()
            ]

        else:
            raise ValueError(
                'Split value {} not supported. Should be in["train", "val", "test"]'.format(
                    split
                )
            )

        # Shuffle data
        random_indices = list(range(len(labels)))
        np.random.RandomState(self.seed).shuffle(
            random_indices
        )  # always shuffle in the same way
        labels = [labels[i] for i in random_indices]
        images = [images[i] for i in random_indices]

        return images, labels

    def get_sample_indices_of_label(self, label):
        """
        Get the indices of the samples that are assigned
        the given label.

        :param label:   Label
        :return:        Indices of samples with the given label
        """
        indices = list(np.nonzero(np.array(self.labels) == label)[0])
        return indices

    def get_filtered_data(self, samples, labels):
        indices = [
            idx
            for (idx, lab) in enumerate(labels)
            if lab.item() in self.mapped_class_to_idx.keys()
        ]
        return samples[indices], labels[indices]
