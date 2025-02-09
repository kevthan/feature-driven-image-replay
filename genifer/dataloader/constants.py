##################################
# Constants for continual learning
##################################

AVAILABLE_DATASETS = ["CIFAR100"]

NORMALIZATION_FACTORS = {
    "CIFAR100": {"mean": (0.507, 0.487, 0.441), "std": (0.267, 0.256, 0.276)}
}

# Input [0, 1] will be mapped to range [-1, 1] by (value - mean) / std
DEFAULT_NORMALIZATION_FACTORS = {
    "CIFAR100": {"mean": (0.507, 0.487, 0.441), "std": (0.267, 0.256, 0.276)}
}

INPUT_DIMENSIONS = {"CIFAR100": [3, 32, 32]}

DATASET_CLASSES = {
    "CIFAR100": {
        "module": "genifer.dataloader.torchvision_loader",
        "class": "TorchVisionDataset",
    }
}

TRAIN_FRACTION = 1.0
