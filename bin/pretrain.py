#!/usr/bin/env python3

import os
import sys
import time

sys.path.append(os.getcwd())

"""
Train model in a class-incremental fashion.
"""
import logging
import json
import importlib
from argparse import ArgumentParser
import numpy as np

import torch

from genifer.utils.io import (
    get_device,
    complete_path,
    set_seed,
    load_checkpoint,
    save_checkpoint,
)
from genifer.utils.misc import flatten_config
from genifer.utils.ml_logging import MLLogger
from genifer.dataloader.task_loader import TaskLoader
from genifer.model.continual_model import ContinualModel
from genifer.learner import Learner

# Reproducibility
torch.backends.cudnn.deterministic = False  # True
torch.backends.cudnn.benchmark = True  # False

# Setup logging
log = logging.getLogger("CL::Supervisor")
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
h1.setFormatter(formatter)
log.addHandler(h1)

# Available trainers
AVAILABLE_TRAINERS = {
    "genifer": ["genifer_trainer", "GeniferTrainer"],
    "pretrain": ["base_trainer", "BaseTrainer"],
}


def pretrain(config_path, run_id, pretrain_mode, clsf_chpt=None):
    """
    Train on sequential tasks using the model, data and
    method provided in the configuration.

    :param config_path:     Path to configuration file
    :param run_id:          ID of this run
    :return:                None
    """
    # Load configuration parameters and construct tasks
    flat_config, config, device, cl_method, task_loaders = prepare(config_path)

    prev_model = None
    train_loader = None
    init_step = 0
    model_id = run_id
    model_path = complete_path(config["misc_params"]["results_path"], model_id)
    ml_logging_mode = config["misc_params"].get("ml_logging", "tensorboard")
    ml_log_dir = "{}_summaries".format(model_path)
    test_metrics = []

    # Setup tensorboard summaries
    ml_logger = MLLogger(mode=ml_logging_mode, log_dir=ml_log_dir)
    ml_logger.start(params=flat_config)

    # Loop over tasks
    for i_t, task_loader in enumerate(task_loaders):
        # pretraining, only T0
        if i_t != 0:
            continue

        # Get train, test, and val loader
        data_loaders = task_loader.get_data_loaders(shuffle=True)
        train_loader = data_loaders["train"]
        test_loaders = data_loaders["test"]

        # Build model
        cl_model = ContinualModel(
            config=config,
            n_classes=task_loader.n_prev_curr_cls,
            device=device,
            prev_model=prev_model,
        )

        # Initialize trainer
        trainer_module = importlib.import_module(
            "genifer.trainers.{}".format(AVAILABLE_TRAINERS[cl_method][0])
        )
        trainer_class = getattr(trainer_module, AVAILABLE_TRAINERS[cl_method][1])
        trainer = trainer_class(
            model=cl_model,
            config=config,
            train_loader=train_loader,
            model_path=model_path,
            ml_logger=ml_logger,
        )

        # Initialize learner
        learner = Learner(
            model=cl_model,
            init_step=init_step,
            config=config,
            model_path=model_path,
            device=device,
            task_id=i_t,
            ml_logger=ml_logger,
        )

        if pretrain_mode == "classifier":
            # train classifier
            log.info("Train on task {}...".format(i_t))
            class_start_time = time.time()
            learner.train(
                train_loader=train_loader, test_loaders=test_loaders, trainer=trainer
            )
            class_duration = time.time() - class_start_time
            log.info(
                "Training classifier on task {} took {:.2f} minutes".format(
                    i_t, class_duration / 60
                )
            )
            # Test
            log.info("Test on seen tasks...")
            trainer.test()
            metrics = learner.test(test_loaders=test_loaders)
            test_metrics.append(metrics)

        else:
            # load classifier checkpoint
            if clsf_chpt is not None:
                load_checkpoint(cl_model, clsf_chpt)
                save_checkpoint(
                    cl_model, "{}_model_best_val_loss.pt".format(model_path)
                )
            # train GAN
            log.info("Training GAN on task {}...".format(i_t))
            gan_start_time = time.time()
            trainer.prepare_next_training(train_loader)
            gan_duration = time.time() - gan_start_time
            log.info(
                "Training GAN on task {} took {:.2f} minutes".format(
                    i_t, gan_duration / 60
                )
            )

    # Save meta data checkpoint
    meta_data = {"test_metrics": test_metrics}
    torch.save(meta_data, "{}_meta_data.pt".format(model_path))

    # Compute and log average test accuracy
    if len(test_metrics) > 0:
        test_accuracies = []
        overall_accuracies = []
        for t_m in test_metrics:
            test_accuracies.extend(
                [
                    t_m[metric_name]
                    for metric_name in t_m.keys()
                    if metric_name.startswith("test_acc_task_")
                ]
            )
            overall_accuracies.append(t_m["overall_sh_test_acc"])
        avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
        if len(overall_accuracies) > 1:
            avg_inc_accuracy = sum(overall_accuracies[1:]) / len(overall_accuracies[1:])
        else:
            avg_inc_accuracy = overall_accuracies[0]
        metric_dict = {
            "avg_test_accuracy": avg_test_accuracy,
            "avg_inc_accuracy": avg_inc_accuracy,
        }
        ml_logger.run(
            func_name="log_metrics", mode="mlflow", metrics=metric_dict, step=init_step
        )

        # Save test metrics as JSON
        test_metrics_json = {"avg_inc_accuracy": avg_inc_accuracy}
        for t_m_dict in test_metrics:
            for key, val in t_m_dict.items():
                if key in test_metrics_json:
                    test_metrics_json[key].append(val)
                else:
                    test_metrics_json[key] = [val]
        test_metrics_fname = "{}_test_metrics.json".format(model_path)
        with open(test_metrics_fname, "w") as t_file:
            json.dump(test_metrics_json, t_file, indent=4)

        # Log as artifact
        ml_logger.run(
            func_name="log_artifact", mode="mlflow", local_path=test_metrics_fname
        )

    # close summary writer
    ml_logger.end()


def prepare(config_path):
    """
    Load configuration parameters and construct task loaders.

    :param config_path:     Path to configuration file
    :return:                - Configuration dictionary
                            - Device
                            - Name of CL method
                            - List of task loaders
    """

    # Load configuration
    with open(config_path, "r") as conf_file:
        config = json.load(conf_file)

    # Log configuration parameters
    flat_config = flatten_config(config)

    # Set seed and CUDA configuration
    set_seed(config["misc_params"]["seed"])
    device = get_device()

    # Generate task split info
    task_split_info = {
        task_id: [] for task_id in range(config["cl_params"]["num_tasks"])
    }
    n_cls_per_task = config["cl_params"]["num_classes_per_task"]

    for task_id in task_split_info.keys():
        indices = sum(n_cls_per_task[:task_id]) + np.array(
            range(n_cls_per_task[task_id])
        )
        if config["cl_params"].get("class_ids_per_task", None) is not None:
            reordered_classes = config["cl_params"]["class_ids_per_task"][task_id]
        else:
            reordered_classes = indices
        task_split_info[task_id] = {
            key: value for key, value in zip(reordered_classes, indices)
        }

    # Get CL method
    cl_method = config["cl_params"]["method"]["name"]

    # Create task loaders
    log.info("Create task loaders...")
    n_tasks = len(task_split_info.keys())
    task_loaders = []
    for task_id in range(n_tasks):
        task_loaders.append(
            TaskLoader(
                split_info=task_split_info,
                task_id=task_id,
                normalization=config["data_params"]["normalization"],
                dataset_name=config["data_params"]["dataset_name"],
                dataset_path=config["data_params"]["dataset_path"],
                device=device,
                batch_size=config["data_params"]["batch_size"],
                seed=config["misc_params"]["seed"],
            )
        )

    return flat_config, config, device, cl_method, task_loaders


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument(
        "--config_path",
        type=str,
        default="../config/config_pretraining_gan.json",
        help="Path to configuration file",
    )
    argparser.add_argument(
        "--id",
        type=str,
        default="genifer",
        help="ID of this run (for models, results etc.)",
    )
    argparser.add_argument(
        "--pretrain_mode",
        type=str,
        default="GAN",
        help="If pretraining, whether to train classifier or GAN",
    )
    argparser.add_argument(
        "--classifier_chpt",
        type=str,
        default=None,
        help="Path to classifier checkpoint (for pretraining feature matching GAN)",
    )
    args = argparser.parse_args()

    assert args.pretrain_mode in ["classifier", "GAN"]

    pretrain(
        config_path=args.config_path,
        run_id=args.id,
        pretrain_mode=args.pretrain_mode,
        clsf_chpt=args.classifier_chpt,
    )
    log.info("Done.")
