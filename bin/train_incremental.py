#!/usr/bin/env python3

import os
import sys
import time
import random

sys.path.append(os.getcwd())

"""
Train model in a class-incremental fashion.
"""

import sys
import logging
import json
import importlib
import os
from argparse import ArgumentParser
from copy import deepcopy
import numpy as np

import torch

from genifer.utils.io import get_device, complete_path, set_seed
from genifer.utils.io import load_checkpoint, save_checkpoint
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
    "pretrain": ["base_trainer", "BaseTrainer"],
    "genifer": ["genifer_trainer", "GeniferTrainer"],
}


def train(config_path, run_id, start_training_with, start_task, prev_task, end_task):
    """
    Train on sequential tasks using the model, data and
    method provided in the configuration.

    :param config_path:         Path to configuration file
    :param run_id:              ID of this run
    :param start_training_with: Start training with 'GAN' or 'classifier'
    :param start_task:          ID of the first task of this sequence
    :param prev_task:           ID of the previous task
    :param end_task:            ID of the last task
    :return:                    None
    """
    # Load configuration parameters and construct tasks
    flat_config, config, device, cl_method, task_loaders = prepare(config_path)

    prev_model = None
    trainer = None
    clsf_step = 0
    model_id = run_id
    model_path = complete_path(config["misc_params"]["results_path"], model_id)
    n_tasks = config["cl_params"]["num_tasks"]
    ml_logging_mode = config["misc_params"].get("ml_logging", "tensorboard")
    ml_log_dir = "{}_summaries".format(model_path)
    test_metrics = []

    # Set up checkpoint paths
    clsf_chpt = config["cl_params"]["method"]["classifier_chpt"]
    if start_task > prev_task:
        ema_g_chpt = config["cl_params"]["method"]["ema_generator_chpt"]
        d_chpt = config["cl_params"]["method"]["d_chpt"]
        meta_chpt = config["cl_params"]["method"].get("meta_chpt", None)
    else:  # load from job chaining checkpoints
        if os.path.isfile("{}_model_best_val_loss.pt".format(model_path)):
            clsf_chpt = "{}_model_best_val_loss.pt".format(model_path)
        ema_g_chpt = "{}_generator.pt".format(model_path)
        d_chpt = "{}_discriminator.pt".format(model_path)
        meta_chpt = "{}_meta_data.pt".format(model_path)

    # Setup tensorboard/mlflow summaries
    ml_logger = MLLogger(mode=ml_logging_mode, log_dir=ml_log_dir)
    if prev_task == 0:
        ml_logger.start(params=flat_config)

    i_t = 0
    while i_t <= end_task:

        # Get train, test, and val loader
        task_loader = task_loaders[i_t]
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

        # Initialize or update trainer
        if trainer is None:
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
        else:
            trainer.update(cl_model, train_loader)

        # Initialize learner
        learner = Learner(
            model=cl_model,
            init_step=clsf_step,
            config=config,
            model_path=model_path,
            device=device,
            task_id=i_t,
            ml_logger=ml_logger,
        )

        # ==========================================================================================
        # Train classifier and/or GAN
        # ==========================================================================================
        if i_t >= prev_task + 1:
            if (i_t > prev_task + 1) or (
                i_t == prev_task + 1 and start_training_with == "classifier"
            ):
                # Train classifier
                log.info("Train on task {}...".format(i_t))
                clsf_start_time = time.time()
                learner.train(
                    train_loader=train_loader,
                    test_loaders=test_loaders,
                    trainer=trainer,
                )
                clsf_duration = time.time() - clsf_start_time
                log.info(
                    "Training classifier on task {} took {:.2f} minutes".format(
                        i_t, clsf_duration / 60.0
                    )
                )

                # Test
                log.info("Test on seen tasks...")
                trainer.test()
                metrics = learner.test(test_loaders=test_loaders)
                test_metrics.append(metrics)
                log.info(
                    "Classifier saved under: {}_model_best_val_loss.pt".format(
                        model_path
                    )
                )

            if n_tasks - 1 > i_t >= prev_task + 1:
                # Load current task classifier (for feature matching)
                if (i_t == prev_task + 1) and (start_training_with == "GAN"):
                    load_checkpoint(cl_model, clsf_chpt)
                    save_checkpoint(
                        cl_model, "{}_model_best_val_loss.pt".format(model_path)
                    )

                # Train GAN
                log.info("Training GAN on task {}...".format(i_t))
                gan_start_time = time.time()
                trainer.prepare_next_training(train_loader)
                gan_duration = time.time() - gan_start_time
                log.info(
                    "Training GAN on task {} took {:.2f} minutes".format(
                        i_t, gan_duration / 60.0
                    )
                )
                log.info("GAN saved under: {}_generator.pt".format(model_path))

            clsf_step = learner.step

            # ======================================================================================
            # Save meta data
            # ======================================================================================
            torch_rng_state = torch.get_rng_state()
            np_rng_state = np.random.get_state()
            python_rng_state = random.getstate()
            meta_data = {
                "test_metrics": test_metrics,
                "mlflow_run_id": ml_logger.run(
                    func_name="active_run", mode="mlflow"
                ).info.run_id,
                "classifier_step": clsf_step,
                "trainer_step": trainer.step,
                "torch_rng_state": torch_rng_state,
                "np_rng_state": np_rng_state,
                "python_rng_state": python_rng_state,
            }
            if get_device().type != "cpu":
                meta_data["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
            torch.save(meta_data, "{}_meta_data.pt".format(model_path))

        prev_model = deepcopy(cl_model)

        # ==========================================================================================
        # Load checkpoints
        # ==========================================================================================
        if i_t == prev_task:
            # Load meta data if available
            if meta_chpt is not None and os.path.isfile(meta_chpt):
                meta_data = torch.load(meta_chpt)
                test_metrics = meta_data["test_metrics"]
                if i_t != 0:
                    ml_logger.run(
                        func_name="start_run",
                        mode="mlflow",
                        run_id=meta_data["mlflow_run_id"],
                    )
                    clsf_step = meta_data["classifier_step"]
                    trainer.step = meta_data["trainer_step"]
                    torch.set_rng_state(meta_data["torch_rng_state"])
                    if get_device().type != "cpu":
                        torch.cuda.set_rng_state(meta_data["torch_cuda_rng_state"])
                    np.random.set_state(meta_data["np_rng_state"])
                    random.setstate(meta_data["python_rng_state"])

            else:
                log.info("No meta data available")

            # Load previous classifier
            if start_training_with == "classifier":
                prev_model = load_checkpoint(prev_model, clsf_chpt)
                save_checkpoint(
                    cl_model, "{}_model_best_val_loss.pt".format(model_path)
                )

            # Load previous GAN
            load_checkpoint(trainer.generator_ema, ema_g_chpt)
            load_checkpoint(trainer.generator, ema_g_chpt)
            load_checkpoint(trainer.discriminator, d_chpt)

            # Set up trainer
            trainer.prev_model = deepcopy(prev_model)
            trainer.prev_model.train()
            trainer.prev_model.requires_grad(False)
            trainer.prev_activations = {}
            for name, module in trainer.prev_model.feature_extractor.named_children():
                module.register_forward_hook(
                    trainer.prev_model.get_activations(name, trainer.prev_activations)
                )

        i_t += 1

    # ==============================================================================================
    # Compute and log test metrics
    # ==============================================================================================
    if len(test_metrics) > 0:
        test_accuracies = []
        overall_accuracies = []
        prev_accuracies = []
        for t_m in test_metrics:
            test_accuracies.extend(
                [
                    t_m[metric_name]
                    for metric_name in t_m.keys()
                    if metric_name.startswith("test_acc_task_")
                ]
            )
            overall_accuracies.append(t_m["overall_sh_test_acc"])
            if "prev_task_acc" in t_m.keys():
                prev_accuracies.append(t_m["prev_task_acc"])
        avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
        if len(overall_accuracies) > 1:
            avg_inc_accuracy = sum(overall_accuracies[1:]) / len(overall_accuracies[1:])
            avg_inc_accuracy_0 = sum(overall_accuracies) / len(overall_accuracies)
        else:
            avg_inc_accuracy = overall_accuracies[0]
            avg_inc_accuracy_0 = avg_inc_accuracy
        if len(prev_accuracies) > 0:
            avg_prev_accuracy = sum(prev_accuracies) / len(prev_accuracies)
        else:
            avg_prev_accuracy = 0

        # Build test metrics dict
        test_metrics_json = {
            "avg_inc_accuracy": avg_inc_accuracy,
            "avg_inc_accuracy_0": avg_inc_accuracy_0,
            "avg_prev_accuracy": avg_prev_accuracy,
        }
        for t_m_dict in test_metrics:
            for key, val in t_m_dict.items():
                if key in test_metrics_json:
                    test_metrics_json[key].append(val)
                else:
                    test_metrics_json[key] = [val]
        curr_accuracies = [
            test_metrics_json[key][0]
            for key in test_metrics_json.keys()
            if key.startswith("test_acc_task_") and key != "test_acc_task_0"
        ]
        if len(curr_accuracies) > 0:
            avg_curr_accuracy = sum(curr_accuracies) / len(curr_accuracies)
        else:
            avg_curr_accuracy = 0
        test_metrics_json["avg_curr_accuracy"] = avg_curr_accuracy

        # Dump JSON
        test_metrics_fname = "{}_test_metrics.json".format(model_path)
        with open(test_metrics_fname, "w") as t_file:
            json.dump(test_metrics_json, t_file, indent=4)

        # Log test metrics as artifact
        metric_dict = {
            "avg_test_accuracy": avg_test_accuracy,
            "avg_inc_accuracy": avg_inc_accuracy,
            "avg_inc_accuracy_0": avg_inc_accuracy_0,
            "avg_curr_accuracy": avg_curr_accuracy,
            "avg_prev_accuracy": avg_prev_accuracy,
        }
        ml_logger.run(
            func_name="log_metrics", mode="mlflow", metrics=metric_dict, step=clsf_step
        )

    # Close ML logger
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
        default="finalTest",
        help="ID of this run (for models, results etc.)",
    )
    argparser.add_argument(
        "--start_training_with",
        type=str,
        choices=["classifier", "GAN"],
        default="classifier",
        help="Start training with classifier or GAN",
    )
    argparser.add_argument(
        "--start_task", type=int, default=1, help="ID of the start task"
    )
    argparser.add_argument(
        "--prev_task", type=int, default=0, help="ID of the previous task"
    )
    argparser.add_argument("--end_task", type=int, default=5, help="ID of the end task")
    args = argparser.parse_args()

    train(
        config_path=args.config_path,
        run_id=args.id,
        start_training_with=args.start_training_with,
        start_task=args.start_task,
        prev_task=args.prev_task,
        end_task=args.end_task,
    )
    log.info("Done.")
