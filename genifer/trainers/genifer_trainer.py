"""
Trainer for generative feature-driven image replay (Genifer).
"""

import sys
import logging
import time
from copy import deepcopy
import importlib

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
from sklearn.metrics import accuracy_score
from torchvision.utils import make_grid

from genifer.dataloader.constants import (
    INPUT_DIMENSIONS,
    DEFAULT_NORMALIZATION_FACTORS,
)
from genifer.trainers.base_trainer import BaseTrainer
from genifer.utils.io import get_device, load_checkpoint, save_checkpoint
from genifer.utils.misc import get_n_trainable_params
import genifer.augmentation.dnnlib.util as dnnlib
import genifer.augmentation.torch_utils.misc as misc

log = logging.getLogger("CL::GeniferTrainer")
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
h1.setFormatter(formatter)
log.addHandler(h1)


def _split_noise_dict(d, n):
    """
    Gathers first n elements form each value in d
    """
    d_out = {}
    for k, v in d.items():
        d_out[k] = v[:n]
    return d_out


class GeniferTrainer(BaseTrainer):

    def __init__(self, model, model_path, config, train_loader, ml_logger):
        super(GeniferTrainer, self).__init__(
            model=model,
            model_path=model_path,
            config=config,
            train_loader=train_loader,
            ml_logger=ml_logger,
        )

        # Load parameters from config
        self.latent_dim = config["cl_params"]["method"].get("latent_dim", 32)
        self.n_tasks = config["cl_params"]["num_tasks"]
        self.n_cls_per_task = config["cl_params"]["num_classes_per_task"]
        self.n_cls = sum(self.n_cls_per_task)
        self.n_iter = config["cl_params"]["method"].get("n_iter", 100)
        self.batch_size = config["cl_params"]["method"].get("batch_size", 128)
        self.disc_lr = config["cl_params"]["method"].get("disc_lr", 0.0002)
        self.gen_lr = config["cl_params"]["method"].get("gen_lr", 0.0002)
        self.prev_curr_ratio = config["cl_params"]["method"].get("prev_curr_ratio", 0.5)
        self.coefficient = config["cl_params"]["method"].get("coefficient", 1.0)
        self.feat_coefficient = config["cl_params"]["method"].get(
            "feat_coefficient", 1.0
        )
        self.out_distill_loss = config["cl_params"]["method"].get(
            "out_distill_loss", "KD"
        )
        self.temperature = config["cl_params"]["method"].get("temperature", 1.0)
        self.feat_matching_layer = config["cl_params"]["method"].get(
            "feat_matching_layer", "stage0"
        )
        self.feat_dist_layer = config["cl_params"]["method"].get(
            "feat_dist_layer", "stage0"
        )
        self.feat_dist_loss_type = config["cl_params"]["method"].get(
            "feat_dist_loss_type", "average"
        )
        self.freeze_layers = config["cl_params"]["method"].get("freeze_layers", False)
        self.matching_mode = config["cl_params"]["method"].get(
            "matching_mode", "indirect"
        )
        self.fake_input_stage = (
            "stage0" if self.matching_mode == "indirect" else self.feat_matching_layer
        )
        self.gan_augmentation = config["cl_params"]["method"].get(
            "gan_augmentation", False
        )
        self.reinit_gen = config["cl_params"]["method"].get("reinit_gen", False)
        self.reinit_disc = config["cl_params"]["method"].get("reinit_disc", False)
        self.device = get_device()

        # get dataset info
        self.dataset_name = config["data_params"]["dataset_name"]
        self.image_channels, self.image_size = INPUT_DIMENSIONS[self.dataset_name][:-1]

        # latent vector for tb summary images palette only from T0 atm
        # we want to do this at the very beginning s.t. we get the same samples for different architectures
        self.n_fixed_z = 100 if self.dataset_name == "CIFAR100" else 16
        for t in range(self.n_tasks):
            setattr(
                self,
                "tb_fixed_z{}".format(t),
                self.sample_z(self.n_fixed_z, mode="fixed", T=t),
            )

        # Build generator model
        self.generator = self._init_generator()
        # EMA half-life in 1k imgs (500k imgs proposed in paper):
        self.ema_nk_img = config["cl_params"]["method"].get("ema_hl", 500)
        # linearly ramp-up EMA horizon from 0 to ema_nk_nkimg in x steps, mitigates bias from doing it right away
        # self.ema_rampup = 0.1953125  # -> 20k steps to linearly reach ema_nk_img from 0
        self.ema_rampup_div = 4
        self.ema_rampup = (
            self.ema_nk_img
            * 1000
            / (self.batch_size * self.n_iter / self.ema_rampup_div * len(train_loader))
        )
        self.generator_ema = deepcopy(self.generator).eval()
        self.generator_ema.requires_grad(False)
        log.info(
            "Number of trainable params in G: {}".format(
                get_n_trainable_params(self.generator)
            )
        )
        self.discriminator = self._init_discriminator()

        log.info(
            "Number of trainable params in D: {}".format(
                get_n_trainable_params(self.discriminator)
            )
        )
        self.prev_generator = None
        self.prev_model = None
        if self.dataset_name == "CIFAR100":
            self.aug_transform_gan = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4, padding_mode="edge"),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            self.aug_transform_gan = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(self.image_size),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        self.step = 0

        # GAN configuration
        self.n_critic = int(config["cl_params"]["method"].get("n_critic", 1))
        # R1 regularization (gp on D(real)) weight: set to 0 for automatic mode
        # 0.01 proposed by paper for cifar10, bs=64
        self.r1_gamma = config["cl_params"]["method"].get("r1_gamma", None)
        if self.r1_gamma is None:
            if self.image_size == 32:
                self.r1_gamma = 0.005
            else:
                gamma_0 = 0.0002
                # would be 0.0016 for cifar with bs=128
                self.r1_gamma = gamma_0 * (self.image_size**2 / self.batch_size)

        # Lazy R1 regularization
        self.r1_interval = config["cl_params"]["method"].get("r1_interval", 16)
        self.lazy_c = 1
        self.d_b1 = 0.0
        self.d_b2 = 0.99
        if self.r1_interval > 1:
            self.lazy_c = self.r1_interval / (self.r1_interval + 1)
            self.disc_lr *= self.lazy_c
            self.d_b1 = self.d_b1**self.lazy_c
            self.d_b2 = self.d_b2**self.lazy_c

        self.gen_coefficient = config["cl_params"]["method"].get("gen_coefficient", 0.0)
        self.g_dist_ada = config["cl_params"]["method"].get("gen_dist_ada", False)
        self.gen_dist_loss_type = config["cl_params"]["method"].get(
            "gen_dist_loss", "L1"
        )
        self.gen_dist_layer = config["cl_params"]["method"].get(
            "gen_dist_layer", "stage0"
        )
        if self.gen_dist_loss_type == "L1":
            self.gen_dist_loss_fn = torch.nn.L1Loss(reduction="mean")
        else:
            self.gen_dist_loss_fn = torch.nn.MSELoss(reduction="mean")

        # D augmentation: p in case of fixed augmentation, target-p for ADA. 0: disable D augmentation
        self.dis_aug_p = config["cl_params"]["method"].get("dis_augmentation_p", 0.0)
        # maximum ADA p value (for clipping)
        self.max_dis_aug_p = config["cl_params"]["method"].get(
            "max_dis_augmentation_p", 0.5
        )
        # adaptive augmentation?
        self.dis_aug_ada = config["cl_params"]["method"].get(
            "dis_augmentation_ada", False
        )
        # how many 1k steps to adjust p from 0 to 1 (paper proposes 500k imgs, would be 4)
        self.ada_nk_iter = config["cl_params"]["method"].get(
            "dis_augmentation_nk_iter", 4
        )

        if self.dis_aug_p > 0:  # augment with p
            # augmentation types proposed in paper as default for new dataset
            specs = dict(
                xflip=1,
                rotate90=1,
                xint=1,
                scale=1,
                rotate=1,
                aniso=1,
                xfrac=1,
                brightness=1,
                contrast=1,
                lumaflip=1,
                hue=1,
                saturation=1,
            )
            aug_kwargs = dnnlib.EasyDict(
                class_name="genifer.augmentation.training.augment.AugmentPipe", **specs
            )
            self.aug_pipe = (
                dnnlib.construct_class_by_name(**aug_kwargs)
                .train()
                .requires_grad_(False)
                .to(self.device)
            )

            # we also evaluate this estimate with fixed p
            # estimate how to adapt every ada_interval steps
            self.ada_interval = 4
            # buffer for moving avg
            self.d_real_sign_accu = [
                torch.as_tensor(0, device=self.device) for _ in range(self.ada_interval)
            ]

            if self.dis_aug_ada:  # adaptive p
                log.info("Augmenting D adaptively using ADA")
                p = 0

            else:  # fixed p
                log.info(
                    "Augmenting D using ADA with fixed p = {}".format(self.dis_aug_p)
                )
                p = self.dis_aug_p

            # set p in aug. pipeline
            self.aug_pipe.p.copy_(torch.as_tensor(p))

        else:
            self.aug_pipe = None

        # Resume GAN training
        self.resume_training = config["cl_params"]["method"].get(
            "resume_training", False
        )
        self.resume_step = config["cl_params"]["method"].get("resume_step", 0)
        self.resume_chpt_prefix = config["cl_params"]["method"].get(
            "resume_chpt_prefix", None
        )
        if self.resume_training:
            log.info("GAN training will resume from step {}.".format(self.resume_step))

        self.freeze_bn = config["cl_params"]["method"].get("freeze_bn", False)

        # Fixed noise vectors for visualization and testing
        self.test_z_size = 2000
        self.test_z = self.sample_z(self.test_z_size)  # all tasks
        # latent vector for tb summary images palette only from T0 atm
        self.tb_fixed_z = self.sample_z(self.n_fixed_z, mode="seen")
        self.pics_per_cls = 20 if self.dataset_name == "CIFAR100" else 5
        self.palette_z = torch.randn(
            self.n_cls_per_task[0] * self.pics_per_cls,
            self.latent_dim,
            device=self.device,
        )

        # Freeze layers from the beginning if pre-trained layers are used
        pretrained = config["model_params"].get("pretrained", False)
        if self.freeze_layers and pretrained and isinstance(pretrained, str):
            self._freeze_model_layers(self.model)

        self.softplus = torch.nn.Softplus()

    def _init_discriminator(self):

        if self.image_size == 32:
            # Small discriminator for 32x32 at stage0
            disc_module = importlib.import_module(
                "genifer.model.conditional_eql_discriminator"
            )
        else:
            # Large discriminator for 128x128 at stage0
            disc_module = importlib.import_module(
                "genifer.model.conditional_eql_large_discriminator"
            )
        disc_class = getattr(disc_module, "CondEqlDiscriminator")
        discriminator = disc_class(
            img_input_channels=self.model.feature_extractor.stage_info[
                self.feat_matching_layer
            ][0],
            img_input_size=self.model.feature_extractor.stage_info[
                self.feat_matching_layer
            ][1],
            device=self.device,
            n_cls=self.n_cls,
            mbdis=True,
        )
        return discriminator

    def _init_generator(self):

        if self.dataset_name == "CIFAR100":
            # Small generator for 32x32 at stage0
            gen_module = importlib.import_module(
                "genifer.model.conditional_eql_style_generator"
            )
        else:
            # Large generator for 128x128 at stage0
            gen_module = importlib.import_module(
                "genifer.model.conditional_eql_style_large_generator"
            )
        gen_class = getattr(gen_module, "CondEqlGenerator")
        generator = gen_class(
            latent_dim=self.latent_dim,
            image_channels=self.model.feature_extractor.stage_info[
                self.feat_matching_layer
            ][0],
            image_size=self.model.feature_extractor.stage_info[
                self.feat_matching_layer
            ][1],
            device=self.device,
            n_cls=self.n_cls,
            matching_mode=self.matching_mode,
            prev_model=None,
        )
        return generator

    def update(self, model, train_loader):

        if self.freeze_layers:
            self._freeze_model_layers(model)

        self.model = model
        self.train_loader = train_loader
        self.task_id += 1

        # Update previous generator (we want the ema copy to become the ground truth renderer for prev. classes)
        self.prev_generator = deepcopy(self.generator_ema)
        self.prev_generator.eval()
        self.prev_generator.requires_grad(False)
        self.tb_fixed_z = self.sample_z(self.n_fixed_z, mode="seen")
        self.palette_z = torch.randn(
            sum(self.n_cls_per_task[: self.task_id + 1]) * self.pics_per_cls,
            self.latent_dim,
            device=self.device,
        )

        # Update EMA ramp-up for tasks > 0
        self.ema_rampup = (
            self.ema_nk_img
            * 1000
            / (self.batch_size * self.n_iter / self.ema_rampup_div * len(train_loader))
        )

        # Reset ADA p
        if self.task_id > 0 and self.aug_pipe is not None:
            self.aug_pipe.p.copy_(torch.as_tensor(0))

    def prepare_next_training(self, data_loader):
        """
        Train generator to produce artificial data with similar logit statistics
        on the current model as the current data.

        :param data_loader:     Training data loader (current data)
        """

        # Save current state of the model
        self.prev_model = deepcopy(self.model)
        # we don't need classifier unless we do feature matching
        if self.feat_matching_layer != "stage0":
            load_checkpoint(
                self.prev_model, "{}_model_best_val_loss.pt".format(self.model_path)
            )
        self.prev_model.eval()
        self.prev_model.requires_grad(False)

        # Set the current generator to the ema generator
        self.generator = deepcopy(self.generator_ema)

        # Freeze model copy and only train the generator
        if self.task_id != 0 and self.reinit_gen:
            log.info("Re-initializing generator...")
            self.generator = self._init_generator()
        self.generator.train()
        self.generator.requires_grad(True)

        # GAN training
        if self.task_id != 0 and self.reinit_disc:
            log.info("Re-initializing discriminator...")
            self.discriminator = self._init_discriminator()

        optimizer_disc = optim.Adam(
            self.discriminator.parameters(),
            lr=self.disc_lr,
            betas=(self.d_b1, self.d_b2),
        )
        optimizer_gen = optim.Adam(
            self.generator.parameters(), lr=self.gen_lr, betas=(0.0, 0.99)
        )

        # Load saved checkpoints if training should be resumed
        if self.resume_training:
            log.info(
                "Load checkpoints to resume training from {}_***.pt".format(
                    self.resume_chpt_prefix
                )
            )
            load_checkpoint(
                self.generator_ema, "{}_generator.pt".format(self.resume_chpt_prefix)
            )
            load_checkpoint(
                self.generator, "{}_curr_generator.pt".format(self.resume_chpt_prefix)
            )
            load_checkpoint(
                self.discriminator,
                "{}_discriminator.pt".format(self.resume_chpt_prefix),
            )
            if self.aug_pipe is not None:
                self.aug_pipe.p.copy_(
                    torch.load("{}_aug_pipe_p.pt".format(self.resume_chpt_prefix))
                )
            optimizer_chpts = torch.load(
                "{}_optimizers.pt".format(self.resume_chpt_prefix)
            )
            optimizer_disc.load_state_dict(optimizer_chpts["optimizer_disc"])
            optimizer_gen.load_state_dict(optimizer_chpts["optimizer_gen"])

        # Define data loader for GAN training (no augmentation, possibly different batch size)
        gan_dataset = deepcopy(data_loader.dataset)
        if self.image_size == 32:
            gan_dataset.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(gan_dataset.mean, gan_dataset.std),
                ]
            )
        else:
            if self.dataset_name == "CIFAR100":
                if self.feat_matching_layer == "stage0":
                    gan_dataset.transform = transforms.Compose(
                        [
                            transforms.Resize(128),
                            transforms.ToTensor(),
                            transforms.Normalize(gan_dataset.mean, gan_dataset.std),
                        ]
                    )
                else:
                    gan_dataset.transform = transforms.Compose(
                        [
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize(gan_dataset.mean, gan_dataset.std),
                        ]
                    )
            else:
                # Not using bounding boxes
                if self.feat_matching_layer == "stage0":
                    gan_dataset.transform = transforms.Compose(
                        [
                            transforms.Resize(128),
                            transforms.RandomCrop(128),
                            transforms.ToTensor(),
                            transforms.Normalize(gan_dataset.mean, gan_dataset.std),
                        ]
                    )
                else:
                    gan_dataset.transform = transforms.Compose(
                        [
                            transforms.Resize(128),
                            transforms.Resize(256),
                            transforms.RandomCrop(256),
                            transforms.ToTensor(),
                            transforms.Normalize(gan_dataset.mean, gan_dataset.std),
                        ]
                    )

        gan_dataloader = DataLoader(
            dataset=gan_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )

        if self.prev_generator is not None:
            self.prev_generator.eval()
            self.prev_generator.requires_grad(False)

        d_steps = 0
        g_steps = 0
        start_time = time.time()
        for epoch in range(self.n_iter):

            # Skip steps when training is resumed from a checkpoint
            if self.resume_training and self.step < self.resume_step:
                g_steps += len(gan_dataloader)
                self.step += len(gan_dataloader)
                if self.step > self.resume_step:
                    log.info("Skipped {} steps!".format(self.step))
                continue

            for i_batch, (real_curr_images, real_curr_labels) in enumerate(
                gan_dataloader
            ):

                # push real images and labels to device
                real_curr_images = real_curr_images.to(self.device)
                real_curr_labels = real_curr_labels.to(self.device)
                real_curr_labels = self._to_one_hot(real_curr_labels)

                # ratio of previous vs current task samples per batch
                ratio = 0.5
                n_prev = int(ratio * self.batch_size)
                n_curr = self.batch_size - n_prev

                ######################
                # Update discriminator
                ######################
                # train D
                self.discriminator.requires_grad(True)
                self.discriminator.zero_grad()
                # freeze G
                self.generator.requires_grad(False)

                # sample D inputs (z used later for G update)
                imgs_r, labels_r, imgs_f, labels_f, z = self.sample_d_inputs(
                    real_curr_images, real_curr_labels, n_prev, n_curr
                )

                # D(real)
                d_real = self.discriminator(imgs_r, labels_r)
                # D(fake)
                d_fake = self.discriminator(imgs_f, labels_f)

                # update ADA
                if self.aug_pipe is not None:
                    self.update_ada(d_real)

                # D loss (non-saturating + R1)
                d_ns_loss = torch.mean(self.softplus(d_fake) + self.softplus(-d_real))
                # (Lazy) R1 regularization (gp on reals)
                d_r1_loss = self._get_r1(imgs_r, labels_r)
                d_loss = d_ns_loss + d_r1_loss

                # Backprop and GD step
                d_loss.backward()
                optimizer_disc.step()

                if d_steps % self.n_critic == 0:

                    ##################
                    # Update generator
                    ##################
                    # train G
                    self.generator.requires_grad(True)
                    self.generator.zero_grad()
                    # freeze D
                    self.discriminator.requires_grad(False)

                    g_dist = (
                        self.prev_generator is not None and self.gen_coefficient > 0
                    )
                    # if GDist, explicitly sample noise maps, else sample on the fly
                    noise_maps = (
                        self.sample_noise_maps(self.batch_size) if g_dist else None
                    )
                    # generate fakes
                    fake_images = self.generator(z, noise=noise_maps)
                    # upsample generated images
                    fake_images = self._resize(fake_images)

                    # ADA
                    imgs_f = self.augment_non_leaking(fake_images)

                    # feat. matching?
                    if not self.discriminator.accepts(imgs_f):
                        imgs_f = self.prev_model(
                            imgs_f, output_stage=self.feat_matching_layer
                        )[-1]

                    # D(fake)
                    d_fake = self.discriminator(imgs_f, labels_f)

                    # GDist
                    g_dist_loss = 0
                    if g_dist:
                        # send half the batch through prev-G and compute GDist on it
                        noise_maps = _split_noise_dict(noise_maps, n_prev)
                        dist_imgs_gt = self.prev_generator(z[:n_prev], noise=noise_maps)
                        # upsample generated images
                        dist_imgs_gt = self._resize(dist_imgs_gt)
                        # non-ADA-augmented fakes for GDist
                        dist_imgs_f = fake_images[:n_prev]
                        # GDist is evaluated in feature space indicated by 'gen_dist_layer' in config
                        if self.gen_dist_layer != "stage0":
                            dist_imgs_gt = self.prev_model(
                                dist_imgs_gt, output_stage=self.gen_dist_layer
                            )[-1]
                            dist_imgs_f = self.prev_model(
                                fake_images[:n_prev], output_stage=self.gen_dist_layer
                            )[-1]

                        # GDist loss
                        dist_imgs_gt = dist_imgs_gt.view(n_prev, -1)
                        dist_imgs_f = dist_imgs_f.view(n_prev, -1)
                        g_dist_loss = self.gen_dist_loss_fn(dist_imgs_gt, dist_imgs_f)

                    g_dist_factor = 1
                    if self.g_dist_ada:
                        # increase GDist factor with increasing number of previous classes
                        g_dist_factor = self.prev_model.n_prev_cls / (
                            self.prev_model.n_cls - self.prev_model.n_prev_cls
                        )

                    # G loss (non-saturating + GDist)
                    g_gan_loss = torch.mean(self.softplus(-d_fake))
                    loss_gen = (
                        g_gan_loss + self.gen_coefficient * g_dist_factor * g_dist_loss
                    )

                    # Backprop and GD step
                    loss_gen.backward()
                    optimizer_gen.step()

                    # update ema copy
                    with torch.autograd.profiler.record_function("G_ema"):
                        # ramp up linearly from 0 to ema_nk_img*1000
                        ema_nimg = min(
                            self.ema_nk_img * 1000,
                            self.batch_size * g_steps * self.ema_rampup,
                        )
                        ema_beta = 0.5 ** (self.batch_size / max(ema_nimg, 1e-8))
                        # lin. interpolate and update
                        for p_ema, p in zip(
                            self.generator_ema.parameters(), self.generator.parameters()
                        ):
                            p_ema.copy_(p.lerp(p_ema, ema_beta))
                        # copy buffers
                        for b_ema, b in zip(
                            self.generator_ema.buffers(), self.generator.buffers()
                        ):
                            b_ema.copy_(b)

                    # log scalars
                    if self.step % 100 == 0:
                        metric_dict = {
                            "L_G_GAN": g_gan_loss.item(),
                            "L_G_dist": g_dist_loss.item() if g_dist else 0,
                            "L_D_GAN": d_ns_loss.item(),
                            "L_G": loss_gen.item(),
                            "L_D": d_loss.item(),
                            "D_real": torch.mean(d_real).item(),
                            "D_fake": torch.mean(d_fake).item(),
                        }
                        self.ml_logger.run(
                            func_name="log_scalars",
                            mode="tb",
                            metric_dict=metric_dict,
                            step=self.step,
                        )

                    # palette summary of generated images
                    if self.step % 1000 == 0:
                        for i in range(self.task_id + 1):
                            z = getattr(self, "tb_fixed_z{}".format(i))
                            self._log_ema_samples(z, i)
                        log.info("epoch {}/{}".format(epoch + 1, self.n_iter))

                    self.step += 1
                    g_steps += 1

                    # save G chpt
                    if self.step % 10000 == 0:
                        duration = time.time() - start_time
                        start_time = time.time()
                        log.info("10k steps took {:.2f} minutes".format(duration / 60))
                        if self.task_id == 0:
                            log.info("Save checkpoint for step {}".format(self.step))
                            save_checkpoint(
                                self.generator,
                                "{}_{}K_curr_generator.pt".format(
                                    self.model_path, int(self.step / 1000)
                                ),
                            )
                            save_checkpoint(
                                self.generator_ema,
                                "{}_{}K_generator.pt".format(
                                    self.model_path, int(self.step / 1000)
                                ),
                            )
                            save_checkpoint(
                                self.discriminator,
                                "{}_{}K_discriminator.pt".format(
                                    self.model_path, int(self.step / 1000)
                                ),
                            )
                            if self.aug_pipe is not None:
                                torch.save(
                                    self.aug_pipe.p,
                                    "{}_{}K_aug_pipe_p.pt".format(
                                        self.model_path, int(self.step / 1000)
                                    ),
                                )
                            torch.save(
                                {
                                    "optimizer_gen": optimizer_gen.state_dict(),
                                    "optimizer_disc": optimizer_disc.state_dict(),
                                },
                                "{}_{}K_optimizers.pt".format(
                                    self.model_path, int(self.step / 1000)
                                ),
                            )

                # Increase D step counter
                d_steps += 1

        # Save final checkpoints
        save_checkpoint(
            self.generator_ema,
            "{}_task{}_generator.pt".format(self.model_path, self.task_id),
        )
        save_checkpoint(self.generator_ema, "{}_generator.pt".format(self.model_path))
        save_checkpoint(
            self.discriminator,
            "{}_task{}_discriminator.pt".format(self.model_path, self.task_id),
        )
        save_checkpoint(
            self.discriminator, "{}_discriminator.pt".format(self.model_path)
        )
        if self.task_id == 0 and self.aug_pipe is not None:
            torch.save(
                self.aug_pipe.p,
                "{}_task{}_aug_pipe_p.pt".format(self.model_path, self.task_id),
            )

        # Plot image palette
        self.plot_img_palette()

    def get_modified_loss(self, samples, labels, curr_loss_fn, step=None, i_batch=None):
        """
        Compute distillation loss on generated samples.

        :param samples:         Batch of real current samples
        :param labels:          Batch of real current labels
        :param curr_loss_fn:    Loss function for real current samples
        :param step:            Training step
        :param i_batch:         Batch index
        :return:                Loss
        """

        if self.prev_generator is not None:

            if self.freeze_bn:
                self.prev_model.eval()
            else:
                self.prev_model.train()

            # Divide batch in previous and current samples
            if self.prev_curr_ratio == "adaptive":
                ratio = self.model.n_prev_cls / self.model.n_cls
            else:
                ratio = self.prev_curr_ratio
            n_prev_samples = int(ratio * len(samples))
            n_curr_samples = len(samples) - n_prev_samples

            # Sample random input vectors and generate fake previous samples
            z = self.sample_z(n_prev_samples, mode="previous")
            fake_prev_samples = self.prev_generator(z)
            if self.feat_matching_layer == "stage0" or self.matching_mode == "indirect":
                mean = DEFAULT_NORMALIZATION_FACTORS[self.dataset_name]["mean"]
                std = DEFAULT_NORMALIZATION_FACTORS[self.dataset_name]["std"]
                mins = torch.tensor(
                    [(0 - mean[i]) / std[i] for i in range(3)], device=self.device
                ).view(1, 3, 1, 1)
                maxs = torch.tensor(
                    [(1 - mean[i]) / std[i] for i in range(3)], device=self.device
                ).view(1, 3, 1, 1)
                fake_prev_samples = torch.min(torch.max(fake_prev_samples, mins), maxs)
                fake_prev_samples = self._augment(fake_prev_samples)

            # Get one batch, half previous, half current samples # TODO: adapt for direct matching
            curr_samples = samples[:n_curr_samples]
            prev_curr_samples = torch.cat([fake_prev_samples, curr_samples], dim=0)

            if self.feat_dist_loss_type.startswith("onlycurr"):
                # curr samples in prev model
                prev_curr_target_logits, _, prev_act = self.prev_model(
                    prev_curr_samples[-n_curr_samples:],
                    input_stage=self.fake_input_stage,
                    output_stage=self.feat_dist_layer,
                )

                # prev features in prev model
                prev_prev_target_logits, _ = self.prev_model(
                    prev_curr_samples[:n_prev_samples],
                    input_stage=self.fake_input_stage,
                )

                # curr samples in curr model
                curr_curr_logits, _, curr_act = self.model(
                    prev_curr_samples[-n_curr_samples:],
                    input_stage=self.fake_input_stage,
                    output_stage=self.feat_dist_layer,
                )

                # freeze feature extractor (no gradients from prev samples)
                for stage_name, stage in self.model.feature_extractor.named_children():
                    for param in stage.parameters():
                        param.requires_grad = False
                    if stage_name == self.feat_dist_layer:
                        break

                # prev features in curr model
                curr_prev_logits, _ = self.model(
                    prev_curr_samples[:n_prev_samples],
                    input_stage=self.fake_input_stage,
                )

                # unfreeze feature extractor
                for stage_name, stage in self.model.feature_extractor.named_children():
                    for param in stage.parameters():
                        param.requires_grad = True
                    if stage_name == self.feat_dist_layer:
                        break
                prev_target_logits = torch.cat(
                    [prev_prev_target_logits, prev_curr_target_logits], dim=0
                )
                logits = torch.cat([curr_prev_logits, curr_curr_logits], dim=0)

            else:

                # Get targeted output probabilities of the previous model and real current labels
                prev_target_logits, _, prev_act = self.prev_model(
                    prev_curr_samples,
                    input_stage=self.fake_input_stage,
                    output_stage=self.feat_dist_layer,
                )

                # Single forward pass
                logits, _, curr_act = self.model(
                    prev_curr_samples,
                    input_stage=self.fake_input_stage,
                    output_stage=self.feat_dist_layer,
                )

            # Compute loss on current samples
            curr_labels = labels[:n_curr_samples]
            prev_labels = torch.argmax(prev_target_logits[:n_prev_samples], dim=1)
            curr_loss = curr_loss_fn(
                logits, torch.cat([prev_labels, curr_labels], dim=0)
            )

            # Compute output distillation loss on previous samples
            if self.out_distill_loss == "KD":
                prev_target_probs = F.softmax(
                    prev_target_logits / self.temperature, dim=1
                )
                pred_log_probs = F.log_softmax(logits / self.temperature, dim=1)[
                    :, : self.model.n_prev_cls
                ]
                out_cl_loss = torch.mean(
                    -torch.sum(prev_target_probs * pred_log_probs, dim=1)
                )
            elif self.out_distill_loss == "KD_sep":
                curr_loss = curr_loss_fn(logits[-n_curr_samples:], curr_labels)
                prev_target_probs = F.softmax(
                    prev_target_logits[:n_prev_samples] / self.temperature, dim=1
                )
                pred_log_probs = F.log_softmax(logits / self.temperature, dim=1)[
                    :n_prev_samples, : self.model.n_prev_cls
                ]
                out_cl_loss = torch.mean(
                    -torch.sum(prev_target_probs * pred_log_probs, dim=1)
                )
            elif self.out_distill_loss == "KL":
                curr_loss = curr_loss_fn(logits[-n_curr_samples:], curr_labels)
                prev_target_logits = prev_target_logits[:n_prev_samples]
                prev_target_probs = torch.zeros(n_prev_samples, self.model.n_cls).to(
                    self.device
                )
                prev_target_probs[:, : self.model.n_prev_cls] = F.softmax(
                    prev_target_logits / self.temperature, dim=1
                )
                prev_target_log_probs = torch.log(prev_target_probs + 1e-6)
                pred_log_probs = F.log_softmax(
                    logits[:n_prev_samples] / self.temperature, dim=1
                )
                out_cl_loss = torch.mean(
                    -torch.sum(
                        prev_target_probs * (pred_log_probs - prev_target_log_probs),
                        dim=1,
                    )
                )
            elif self.out_distill_loss == "MSE_prev":
                # Loss only on previous logits, current logits not constrained (set to zero)
                mse = torch.nn.MSELoss(reduction="mean")
                out_cl_loss = mse(
                    prev_target_logits, logits[:n_prev_samples, : self.model.n_prev_cls]
                )
            else:
                # L1
                l1 = torch.nn.L1Loss(reduction="mean")
                out_cl_loss = l1(
                    prev_target_logits, logits[:n_prev_samples, : self.model.n_prev_cls]
                )

            # Compute feature distillation loss on previous samples
            feat_loss = torch.tensor(0.0, device=self.device)
            if self.feat_coefficient > 0:
                if self.feat_dist_loss_type.startswith("curr"):
                    prev_act = prev_act[-n_curr_samples:].view(n_curr_samples, -1)
                    curr_act = curr_act[-n_curr_samples:].view(n_curr_samples, -1)
                elif self.feat_dist_loss_type.startswith("onlycurr"):
                    prev_act = prev_act.view(n_curr_samples, -1)
                    curr_act = curr_act.view(n_curr_samples, -1)
                else:
                    prev_act = prev_act[:n_prev_samples].view(n_prev_samples, -1)
                    curr_act = curr_act[:n_prev_samples].view(n_prev_samples, -1)
                if "single_L1" in self.feat_dist_loss_type:
                    feat_loss = torch.norm(curr_act - prev_act, p=1, dim=1).mean()
                else:
                    feat_loss = torch.norm(curr_act - prev_act, p=2, dim=1).mean()

            return curr_loss, (
                self.coefficient * out_cl_loss,
                self.feat_coefficient * feat_loss,
            )

        else:
            # No CL loss in the first task
            logits = self.model(samples)[0]
            curr_loss = curr_loss_fn(logits, labels)
            cl_loss = None
            return curr_loss, cl_loss

    def test(self):
        """
        Get test accuracy on fake samples.

        :return: None
        """

        if self.prev_generator is not None:
            self.model.eval()
            self.prev_model.eval()

            chunk_idx = 0
            labels = np.zeros(self.test_z_size)
            pred_labels = np.zeros(self.test_z_size)
            while chunk_idx < self.test_z_size:
                end_idx = min(chunk_idx + self.batch_size, self.test_z_size)
                z = self.test_z[chunk_idx:end_idx]
                fake_images = self.prev_generator(z)
                # upsample generated images
                fake_images = self._resize(fake_images)
                if (
                    self.feat_matching_layer == "stage0"
                    or self.matching_mode == "indirect"
                ):
                    fake_images = torch.clamp(fake_images, min=-1.0, max=1.0)
                labels[chunk_idx:end_idx] = (
                    torch.argmax(
                        self.prev_model(fake_images, input_stage=self.fake_input_stage)[
                            0
                        ],
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                )
                pred_labels[chunk_idx:end_idx] = (
                    torch.argmax(
                        self.model(fake_images, input_stage=self.fake_input_stage)[0],
                        dim=1,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                chunk_idx += self.batch_size

            fake_test_acc = accuracy_score(y_true=labels, y_pred=pred_labels)
            log.info("Test accuracy on fake images: {}".format(fake_test_acc))
            metric_dict = {"fake_test_acc": fake_test_acc}
            self.ml_logger.run(
                func_name="log_scalars",
                mode="tb",
                metric_dict=metric_dict,
                step=self.step,
            )
            self.ml_logger.run(
                func_name="log_metrics",
                mode="mlflow",
                metrics=metric_dict,
                step=self.step,
            )

    def sample_latent(self, n):
        return torch.randn((n, self.latent_dim), device=self.device)

    def sample_z(self, n_samples, labels=None, mode=None, T=0):
        """
        Sample a batch of noise vectors and concatenate with one-hot encoded task labels
        if desired.

        :param n_samples:           Number of samples
        :param labels:              Batch of real labels as one-hot
        :param mode:                Sampling mode
        :return:                    Batch of sampled latent vectors concatenated with sampled onehot labels
        """

        # Sample latent
        z = torch.randn(n_samples, self.latent_dim, device=self.device)

        if mode == "fixed":
            # sample random current labels
            labels = torch.randint(
                low=sum(self.n_cls_per_task[:T]),
                high=sum(self.n_cls_per_task[: T + 1]),
                size=(n_samples,),
            )
            one_hot = self._to_one_hot(labels)
            return torch.cat([z, one_hot], dim=1)

        # those that G has seen so far
        if mode == "seen":
            # sample random current labels
            labels = torch.randint(
                low=0,
                high=sum(self.n_cls_per_task[: self.task_id + 1]),
                size=(n_samples,),
            )
            one_hot = self._to_one_hot(labels)
            return torch.cat([z, one_hot], dim=1)

        if labels is None:
            if mode == "previous":
                # randomly sampled labels from previous tasks
                sampled_prev_labels = torch.randint(
                    low=0,
                    high=sum(self.n_cls_per_task[: self.task_id]),
                    size=(n_samples,),
                )
                one_hot_labels = self._to_one_hot(sampled_prev_labels)
            elif mode == "current":
                # labels only from current task
                sampled_curr_labels = torch.randint(
                    low=sum(self.n_cls_per_task[: self.task_id]),
                    high=sum(self.n_cls_per_task[: self.task_id + 1]),
                    size=(n_samples,),
                )
                one_hot_labels = self._to_one_hot(sampled_curr_labels)
            else:
                # labels from all tasks, even 'future' ones (for testing)
                sampled_labels = torch.randint(
                    low=0, high=sum(self.n_cls_per_task), size=(n_samples,)
                )
                one_hot_labels = self._to_one_hot(sampled_labels)

        else:
            # simply return provided real current task labels
            one_hot_labels = labels

        # concat latent and onehot
        return torch.cat([z, one_hot_labels], dim=1)

    def sample_noise_maps(self, n_samples):

        if self.dataset_name == "CIFAR100":
            noise_maps = {
                1: torch.randn(n_samples, 1, 4, 4, device=self.device),
                2: torch.randn(n_samples, 1, 4, 4, device=self.device),
                3: torch.randn(n_samples, 1, 8, 8, device=self.device),
                4: torch.randn(n_samples, 1, 8, 8, device=self.device),
                5: torch.randn(n_samples, 1, 16, 16, device=self.device),
                6: torch.randn(n_samples, 1, 16, 16, device=self.device),
                7: torch.randn(n_samples, 1, 32, 32, device=self.device),
            }
        else:
            noise_maps = {
                1: torch.randn(n_samples, 1, 4, 4, device=self.device),
                2: torch.randn(n_samples, 1, 4, 4, device=self.device),
                3: torch.randn(n_samples, 1, 8, 8, device=self.device),
                4: torch.randn(n_samples, 1, 16, 16, device=self.device),
                5: torch.randn(n_samples, 1, 32, 32, device=self.device),
                6: torch.randn(n_samples, 1, 64, 64, device=self.device),
                7: torch.randn(n_samples, 1, 128, 128, device=self.device),
                8: torch.randn(n_samples, 1, 128, 128, device=self.device),
            }
        return noise_maps

    def set_generator(self, generator):
        self.generator = generator

    def set_prev_model(self, model):
        self.prev_model = model

    def _get_r1(self, real_img, real_labels):

        # Lazy R1 regularization: don't compute for every mini-batch
        if self.step % self.r1_interval == 0:
            # gradients wrt. img
            real_img = real_img.detach().requires_grad_(True)
            real_logits = self.discriminator(real_img, real_labels)
            r1_grad = torch.autograd.grad(
                outputs=[real_logits.sum()],
                inputs=[real_img],
                create_graph=True,
                only_inputs=True,
            )[0]
            if (self.image_size == 32 and self.feat_matching_layer == "stage5") or (
                self.image_size == 224 and self.feat_matching_layer == "stage6"
            ):
                r1_penalty = r1_grad.square().sum(dim=1)
            else:
                r1_penalty = r1_grad.square().sum(dim=[1, 2, 3])
            r1_loss = r1_penalty.mean() * (self.r1_gamma / 2) * self.lazy_c

            # log r1
            if self.r1_interval > 1 or (self.r1_interval == 1 and self.step % 100 == 0):
                metric_dict = {"d_r1": r1_loss.item()}
                self.ml_logger.run(
                    func_name="log_scalars",
                    mode="tb",
                    metric_dict=metric_dict,
                    step=self.step,
                )
                self.ml_logger.run(
                    func_name="log_metrics",
                    mode="mlflow",
                    metrics=metric_dict,
                    step=self.step,
                )
        else:
            r1_loss = 0.0

        return r1_loss

    def _freeze_model_layers(self, model):
        for stage_name, stage in model.feature_extractor.named_children():
            for param in stage.parameters():
                param.requires_grad = False
            if stage_name == self.feat_dist_layer:
                break

    def _augment(self, images):

        if self.gan_augmentation:
            images = self.aug_transform_gan(images)
        elif self.dataset_name != "CIFAR100":
            aug_transform_gan = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(self.image_size)]
            )
            images = aug_transform_gan(images)

        return images

    def _resize(self, images):

        if self.feat_matching_layer == "stage0":
            if self.dataset_name == "CIFAR100":
                pass
        else:
            if self.dataset_name == "CIFAR100":
                pass
            else:
                # upsample generated 128x128 images to 256x256
                images = transforms.Resize(256)(images)
        return images

    def _to_one_hot(self, labels):

        n_samples = len(labels)
        one_hot_encoding = np.zeros((n_samples, self.n_cls), dtype=np.float32)
        for i_label, label in enumerate(labels):
            one_hot_encoding[i_label, label] = 1.0

        return torch.tensor(one_hot_encoding, dtype=torch.float32, device=self.device)

    def augment_non_leaking(self, img):
        if self.aug_pipe is not None:
            # augment
            img = self.aug_pipe(img)
        return img

    def _log_ema_samples(self, z, T):
        imgs = self.generator_ema(z)
        imgs = self.prepare_for_plotting(imgs)
        n_imgs_per_side = int(np.sqrt(self.n_fixed_z))
        imgs = make_grid(imgs, n_imgs_per_side, n_imgs_per_side)
        self.ml_logger.run(
            func_name="add_image",
            mode="tb",
            tag="fake_imgs_T{}".format(T),
            img_tensor=imgs,
            global_step=self.step,
        )

    def sample_d_inputs(self, curr_images, curr_labels, n_prev, n_curr):
        # real D input
        if self.prev_generator is None:
            imgs_r = curr_images
            labels_r = curr_labels
            z = self.sample_z(self.batch_size, mode="current")
        else:
            # divide batch in previous (pseudo-real) and current (real) task samples
            z_prev = self.sample_z(n_prev, mode="previous")
            prev_imgs = self.prev_generator(z_prev).detach()
            # upsample generated images
            prev_imgs = self._resize(prev_imgs)
            # combine previous (pseudo-real) and current (real) to form real batch for D
            imgs_r = torch.cat([prev_imgs, curr_images[:n_curr]], dim=0)
            labels_r = torch.cat(
                [z_prev[:, self.latent_dim :], curr_labels[:n_curr]], dim=0
            )
            z_curr = self.sample_z(n_curr, mode="current")
            z = torch.cat([z_prev, z_curr], dim=0)

        # fake D input
        imgs_f = self.generator(z).detach()
        # upsample generated images
        imgs_f = self._resize(imgs_f)
        labels_f = z[:, self.latent_dim :]

        # ADA
        imgs_r = self.augment_non_leaking(imgs_r)
        imgs_f = self.augment_non_leaking(imgs_f)

        # feat. matching?
        if not self.discriminator.accepts(imgs_r):
            imgs_r = self.prev_model(imgs_r, output_stage=self.feat_matching_layer)[-1]
            imgs_f = self.prev_model(imgs_f, output_stage=self.feat_matching_layer)[-1]

        return imgs_r, labels_r, imgs_f, labels_f, z

    def update_ada(self, d_real):
        # add mean signs of batch to buffer
        self.d_real_sign_accu.append(torch.mean(torch.sign(d_real)))
        # remove oldest measurement from buffer
        del self.d_real_sign_accu[0]
        # estimate r_t
        r_t = torch.mean(torch.stack(self.d_real_sign_accu))
        if self.dis_aug_ada and self.step % self.ada_interval == 0:
            # adapt augmentation p
            adjust = (
                torch.sign(r_t - self.dis_aug_p)
                * self.ada_interval
                / (self.ada_nk_iter * 1000)
            )
            new_p = (
                (self.aug_pipe.p + adjust)
                .max(misc.constant(0, device=self.device))
                .min(misc.constant(self.max_dis_aug_p, device=self.device))
            )
            self.aug_pipe.p.copy_(new_p)
        # log
        if self.step % 100 == 0:
            self.ml_logger.run(
                func_name="log_scalars",
                mode="tb",
                metric_dict={"ADA_r_t": r_t, "ADA_p": self.aug_pipe.p},
                step=self.step,
            )
            self.ml_logger.run(
                func_name="log_metrics",
                mode="mlflow",
                metrics={"ADA_r_t": r_t.item(), "ADA_p": self.aug_pipe.p.item()},
                step=self.step,
            )

    def plot_img_palette(self):

        # Random vectors with labels
        plot_n_cls = sum(self.n_cls_per_task[: self.task_id + 1])
        plot_labels = [[i for i in range(plot_n_cls)] for j in range(self.pics_per_cls)]
        plot_labels = np.array(plot_labels).flatten()
        plot_one_hot = self._to_one_hot(plot_labels).detach()
        gen_input = torch.cat([self.palette_z, plot_one_hot], dim=1)

        for g_type in ["G_ema", "G_curr"]:
            # Generate images in chunks
            chunk_idx = 0
            n_imgs = gen_input.shape[0]
            plot_imgs = []
            while chunk_idx < n_imgs:
                end_idx = min(chunk_idx + self.batch_size, n_imgs)
                z = gen_input[chunk_idx:end_idx]
                if g_type == "G_ema":
                    imgs = self.generator_ema(z).detach().cpu()
                else:
                    imgs = self.generator(z).detach().cpu()
                plot_imgs.append(imgs)
                chunk_idx += self.batch_size
            plot_imgs = torch.cat(plot_imgs, dim=0)
            plot_imgs = self.prepare_for_plotting(plot_imgs)
            img_palette = make_grid(plot_imgs, plot_n_cls, padding=1)
            img_palette = np.transpose(img_palette.numpy(), axes=(1, 2, 0))

            # Save palette
            img_palette = np.round(img_palette * 255).astype(np.uint8)
            if self.image_size == 32:
                plt.grid(False)
                plt.imsave(
                    "{}_task{}_palette_{}.png".format(
                        self.model_path, self.task_id, g_type
                    ),
                    img_palette,
                )
            else:
                img_palette = Image.fromarray(img_palette)
                img_palette.reduce(4)
                img_palette.save(
                    "{}_task{}_palette_{}.png".format(
                        self.model_path, self.task_id, g_type
                    )
                )

    def prepare_for_plotting(self, imgs):

        n_imgs = len(imgs)
        data_device = "cuda:0" if imgs.is_cuda else "cpu"
        # 'De-normalize' images and map back to range [0,1] for plotting
        mean = DEFAULT_NORMALIZATION_FACTORS[self.dataset_name]["mean"]
        mean = torch.tensor(n_imgs * [mean], device=data_device).view(n_imgs, 3, 1, 1)
        std = DEFAULT_NORMALIZATION_FACTORS[self.dataset_name]["std"]
        std = torch.tensor(n_imgs * [std], device=data_device).view(n_imgs, 3, 1, 1)
        imgs = imgs * std + mean
        imgs = torch.clip(imgs, 0.0, 1.0)
        return imgs
