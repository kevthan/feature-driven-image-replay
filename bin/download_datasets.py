import argparse

import torchvision.datasets as datasets

from genifer.dataloader.constants import AVAILABLE_DATASETS


def download_datasets(data_path):
    """
    Download all available datasets to the given location.

    :param data_path:   Path to the directory where the datasets will be stored
    :return:            None
    """

    for dataset_name in AVAILABLE_DATASETS:
        getattr(datasets, dataset_name)(root=data_path, download=True)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the directory where the datasets should be stored",
    )
    args = argparser.parse_args()

    download_datasets(args.dataset_path)
    print("Download complete!")
