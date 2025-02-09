"""This class provides methods to train and test a model."""

import sys
import logging

import torch
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    MultiStepLR,
    LambdaLR,
    ExponentialLR,
)
from sklearn.metrics import accuracy_score

from genifer.utils.io import save_checkpoint, load_checkpoint
from genifer.utils.radam import RAdam


log = logging.getLogger("CL::Learner")
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
h1.setFormatter(formatter)
log.addHandler(h1)


class Learner:

    def __init__(
        self, model, init_step, config, model_path, device, task_id, ml_logger
    ):
        self.model = model
        self.step = init_step
        self.config = config
        self.learning_rate = config["train_params"]["learning_rate"]
        self.final_lr = config["train_params"].get("final_learning_rate", 0.01)
        self.n_epochs = config["train_params"]["n_epochs"]
        self.optimizer = config["train_params"]["optimizer"]
        self.use_scheduler = config["train_params"].get("use_scheduler", False)
        self.scheduler_type = config["train_params"].get("scheduler", "multistep")
        self.scheduler_factor = config["train_params"].get("scheduler_factor", 0.1)
        self.scheduler_milestones = config["train_params"].get(
            "scheduler_milestones", None
        )
        self.weight_decay = config["train_params"].get("weight_decay", 0.0005)
        self.is_upperbound = config["cl_params"]["method"]["name"] == "UpperBound"
        self.ce_coefficient = config["cl_params"].get("ce_coefficient", 1.0)
        self.model_path = model_path
        self.device = device
        self.task_id = task_id
        self.num_tasks = config["cl_params"]["num_tasks"]
        self.ml_logger = ml_logger

    def train(self, train_loader, test_loaders, trainer):
        """
        Train the model.

        :param train_loader:    Training data loader
        :param test_loaders:    Test data loaders
        :param trainer:         Trainer (for different CL methods)
        :return:                None
        """

        # Set trainable parameters single-head training
        trainable_params = self.model.parameters()

        # Define optimizer and loss
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=True,
            )
        elif self.optimizer == "RAdam":
            optimizer = RAdam(
                trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay
            )

        else:
            raise ValueError("Optimizer '{}' not implemented".format(self.optimizer))
        log.info("Using {} optimizer".format(self.optimizer))
        if self.use_scheduler:
            if self.scheduler_type == "plateau":
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.scheduler_factor,
                    patience=10,
                    threshold=0.1,
                    threshold_mode="abs",
                    min_lr=0.0001,
                )
            elif self.scheduler_type == "multistep":
                if self.scheduler_milestones is None:
                    milestones = [
                        int(0.3 * self.n_epochs),
                        int(0.6 * self.n_epochs),
                        int(0.8 * self.n_epochs),
                    ]
                else:
                    milestones = self.scheduler_milestones
                scheduler = MultiStepLR(
                    optimizer, milestones=milestones, gamma=self.scheduler_factor
                )
            elif self.scheduler_type == "lambda":
                scheduler = LambdaLR(
                    optimizer,
                    lr_lambda=lambda epoch: 1 - 0.9 * epoch / (self.n_epochs - 1),
                )
            else:
                scheduler = ExponentialLR(optimizer, gamma=self.scheduler_factor)

        loss_fn = torch.nn.CrossEntropyLoss()

        batch_ctr = 0
        for epoch in range(self.n_epochs):

            # Training
            self.model.train()

            batch_ce_loss = []
            batch_cl_loss = []
            batch_cl_reg_loss = []
            batch_pure_cl_loss = []  # without coefficient
            batch_pure_cl_reg_loss = []  # without coefficient
            for i_b, (samples, labels) in enumerate(train_loader, 1):

                samples = samples.to(self.device)
                labels = labels.to(self.device)
                if not (epoch == 0 and i_b == 1):
                    batch_ctr += 1

                # Get losses
                ce_loss, cl_loss = trainer.get_modified_loss(
                    samples, labels, loss_fn, step=self.step, i_batch=batch_ctr
                )

                if cl_loss is None:
                    cl_reg_loss = None
                    total_loss = ce_loss
                else:
                    if isinstance(cl_loss, tuple):
                        cl_reg_loss = cl_loss[1]
                        cl_loss = cl_loss[0]
                        total_loss = (
                            self.ce_coefficient * ce_loss + cl_loss + cl_reg_loss
                        )
                    else:
                        cl_reg_loss = None
                        total_loss = self.ce_coefficient * ce_loss + cl_loss

                # Backpropagate
                optimizer.zero_grad()
                total_loss.backward()

                if not self.config["train_params"]["train_feature_extractor"]:
                    self.model.feature_extractor.zero_grad()

                # Compute gradient norm
                grad_norms = []
                param_norms = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norms.append(
                            torch.norm(p.grad.detach(), p=2).to(self.device)
                        )
                    param_norms.append(torch.norm(p.detach(), p=2).to(self.device))
                total_grad_norm = torch.norm(torch.stack(grad_norms), p=2)
                total_param_norm = torch.norm(torch.stack(param_norms), p=2)

                # Take optimization step
                optimizer.step()

                # Execute operations after each batch
                trainer.batch_finished(ce_loss)

                # Log training losses
                batch_ce_loss.append(ce_loss.item())
                if cl_loss is None:
                    batch_cl_loss.append(0.0)
                    batch_pure_cl_loss.append(0.0)
                else:
                    batch_cl_loss.append(cl_loss.item())
                    batch_pure_cl_loss.append(
                        0.0
                        if trainer.coefficient == 0
                        else cl_loss.item() / trainer.coefficient
                    )

                if cl_reg_loss is None:
                    batch_cl_reg_loss.append(0.0)
                    batch_pure_cl_reg_loss.append(0.0)
                else:
                    batch_cl_reg_loss.append(cl_reg_loss.item())
                    batch_pure_cl_reg_loss.append(
                        0.0
                        if trainer.feat_coefficient == 0
                        else cl_reg_loss.item() / trainer.feat_coefficient
                    )

                batch_grad_norm = total_grad_norm.item()
                batch_param_norm = total_param_norm.item()

                self.step += 1

                if self.step % 20 == 0:
                    metric_dict = {
                        "batch_ce_loss": batch_ce_loss[-1],
                        "batch_cl_loss": batch_cl_loss[-1],
                        "batch_cl_reg_loss": batch_cl_reg_loss[-1],
                        "batch_pure_cl_loss": batch_pure_cl_loss[-1],
                        "batch_pure_cl_reg_loss": batch_pure_cl_reg_loss[-1],
                        "ce_cl_ratio": (
                            batch_ce_loss[-1] / batch_cl_loss[-1]
                            if batch_cl_loss[-1] != 0
                            else 0
                        ),
                        "pure_ce_cl_ratio": (
                            batch_ce_loss[-1] / batch_pure_cl_loss[-1]
                            if batch_pure_cl_loss[-1] != 0
                            else 0
                        ),
                        "batch_grad_norm": batch_grad_norm,
                        "batch_param_norm": batch_param_norm,
                    }
                    self.ml_logger.run(
                        func_name="log_metrics",
                        mode="mlflow",
                        metrics=metric_dict,
                        step=self.step,
                    )

            # Log training loss per epoch
            avg_batch_ce_loss = sum(batch_ce_loss) / len(batch_ce_loss)
            avg_batch_cl_loss = sum(batch_cl_loss) / len(batch_cl_loss)
            avg_batch_cl_reg_loss = sum(batch_cl_reg_loss) / len(batch_cl_reg_loss)
            avg_batch_pure_cl_loss = sum(batch_pure_cl_loss) / len(batch_pure_cl_loss)

            # Test during training phase
            self.model.eval()

            if (epoch + 1) % 5 == 0:

                all_test_labels = []
                all_test_logits = []
                all_test_predictions = []
                test_metric_dict = {}

                with torch.no_grad():

                    for i_set, test_loader in enumerate(test_loaders):

                        task_test_labels = []
                        task_test_pred = []
                        for samples, labels in test_loader:

                            samples = samples.to(self.device)
                            labels = labels.to(self.device)

                            # Forward pass
                            logits, latent = self.model(samples)
                            pred = torch.argmax(logits, dim=1)

                            all_test_labels.append(labels)
                            all_test_logits.append(logits)
                            task_test_labels.append(labels)
                            task_test_pred.append(pred)
                            all_test_predictions.append(pred)

                        task_test_labels = torch.cat(task_test_labels).cpu().numpy()
                        task_test_pred = torch.cat(task_test_pred).cpu().numpy()
                        test_metric_dict["tr_test_acc_{}".format(i_set)] = (
                            accuracy_score(task_test_labels, task_test_pred)
                        )

                    all_test_labels = torch.cat(all_test_labels)
                    all_test_logits = torch.cat(all_test_logits)
                    all_test_predictions = torch.cat(all_test_predictions)

                    # Log test metrics
                    test_loss = loss_fn(all_test_logits, all_test_labels).item()
                    test_acc = accuracy_score(
                        all_test_labels.cpu().numpy(),
                        all_test_predictions.cpu().numpy(),
                    )
                    test_metric_dict.update(
                        {
                            "tr_test_loss": test_loss,
                            "tr_test_accuracy": test_acc,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )
                    self.ml_logger.run(
                        func_name="log_metrics",
                        mode="mlflow",
                        metrics=test_metric_dict,
                        step=self.step,
                    )

                    log.info(
                        "Epoch {} / {} | Test acc {:.3f} | CE {:.3f} | Pure CL {:.3f} | Weighted CL {:.3f}".format(
                            epoch + 1,
                            self.n_epochs,
                            test_acc,
                            avg_batch_ce_loss,
                            avg_batch_pure_cl_loss,
                            avg_batch_cl_loss,
                        )
                    )

            save_checkpoint(
                self.model, "{}_model_best_val_loss.pt".format(self.model_path)
            )
            save_checkpoint(
                self.model,
                "{}_task{}_model_best_val_loss.pt".format(
                    self.model_path, self.task_id
                ),
            )

            # Update step for learning rate scheduler
            if self.use_scheduler:
                if self.scheduler_type == "plateau":
                    scheduler.step(avg_batch_cl_loss + avg_batch_cl_reg_loss)
                else:
                    scheduler.step()

    def test(self, test_loaders, reload_chpt=True):

        # Load best model
        if reload_chpt:
            load_checkpoint(
                self.model, "{}_model_best_val_loss.pt".format(self.model_path)
            )

        # Test
        self.model.eval()
        test_metrics = {}

        with torch.no_grad():

            # Loop over each task test set
            all_labels = []
            all_probabilities = []
            all_sh_predictions = []
            for i_set, test_loader in enumerate(test_loaders):

                test_labels = []
                test_predictions = []

                for samples, labels in test_loader:

                    samples = samples.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    logits, latent = self.model(samples)
                    probabilities = torch.softmax(logits, dim=1)

                    # Single-head testing
                    sh_predictions = torch.argmax(logits, dim=1)

                    test_labels.append(labels)
                    all_labels.append(labels)
                    all_probabilities.append(probabilities)
                    test_predictions.append(sh_predictions)
                    all_sh_predictions.append(sh_predictions)

                test_labels = torch.cat(test_labels).cpu().numpy()
                test_predictions = torch.cat(test_predictions).cpu().numpy()

                # Compute metrics per task test set
                test_acc = accuracy_score(test_labels, test_predictions)
                test_metrics["test_acc_task_{}".format(i_set)] = test_acc
                log.info("Test accuracy {}: {}".format(i_set, test_acc))

            # Compute overall test accuracy (all classes seen so far)
            all_labels = torch.cat(all_labels).cpu().numpy()
            all_sh_predictions = torch.cat(all_sh_predictions).cpu().numpy()
            overall_sh_test_acc = accuracy_score(all_labels, all_sh_predictions)

            # Log metrics
            test_metrics["overall_sh_test_acc"] = overall_sh_test_acc
            if self.task_id > 0:
                prev_test_acc = accuracy_score(
                    all_labels[: -len(test_labels)],
                    all_sh_predictions[: -len(test_labels)],
                )
                test_metrics["prev_task_acc"] = prev_test_acc

            log.info(
                "Overall single-head test accuracy: {}".format(overall_sh_test_acc)
            )
            self.ml_logger.run(
                func_name="log_metrics",
                mode="mlflow",
                metrics=test_metrics,
                step=self.step,
            )

        return test_metrics
