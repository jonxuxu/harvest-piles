#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import math
import wandb
import psutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    ToPILImage,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
import matplotlib.pyplot as plt
from accelerate import Accelerator

from transformers import Swinv2Config, Swinv2ForMaskedImageModeling
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

from transformers.utils import check_min_version

from config import Swin_Pretrain
from dataset import create_SkysatUnlabelled_dataset
from eval_metrics import get_similarity_metric

""" Pre-training a 🤗 Transformers model for simple masked image modeling (SimMIM).
Any model supported by the AutoModelForMaskedImageModeling API can be used.
"""

# -----------------
# CONFIG
# -----------------
config = Swin_Pretrain()
os.makedirs(config.output_path, exist_ok=True)

pretrained_model_path = "microsoft/swinv2-base-patch4-window8-256"
model_config = Swinv2Config.from_pretrained(pretrained_model_path)

with open(os.path.join(config.output_path, "README.md"), "w+") as f:
    print(config.__dict__, file=f)

# Seed
torch.manual_seed(config.seed)
np.random.seed(config.seed)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.30.0.dev0")

accelerator = Accelerator(
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="wandb",
    mixed_precision=config.mixed_precision,
)
device = accelerator.device

# Log on each process the small summary:
if accelerator.is_main_process:
    print(f"Training/evaluation parameters:")
    print(config.__dict__)

# -----------------
# LOGGER
# -----------------
process = psutil.Process()
if accelerator.is_main_process:
    print(process.memory_info().rss / 1024**3, "GB memory used")

accelerator.init_trackers(
    config.wandb_project,
    config=config,
    init_kwargs={
        "wandb": {
            "group": config.wandb_group,
            "reinit": True,
            "dir": os.path.join(config.working_dir),
        }
    },
)

# -----------------
# DATASET
# -----------------


class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (image_size / model_patch_size)**2 where the value is either 0 or 1,
    where 1 indicates "masked".
    """

    def __init__(
        self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6
    ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError("Input size must be divisible by mask patch size")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError("Mask patch size must be divisible by model patch size")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten())


# create mask generator
mask_generator = MaskGenerator(
    input_size=model_config.image_size,
    mask_patch_size=config.mask_patch_size,
    model_patch_size=config.model_patch_size,
    mask_ratio=config.mask_ratio,
)

# transformations as done in original SimMIM paper
# source: https://github.com/microsoft/SimMIM/blob/main/data/data_simmim.py
transforms = Compose(
    [
        ToPILImage(),
        Resize((model_config.image_size, model_config.image_size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
        Normalize(
            mean=[0.412, 0.368, 0.326], std=[0.110, 0.097, 0.098]
        ),  # our dataset vals
    ]
)


def preprocess_images(x):
    """Preprocess a batch of images by applying transforms + creating a corresponding mask, indicating
    which patches to mask."""
    out = {
        "pixel_values": transforms(x),
        "mask": mask_generator(),
    }
    return out


# Initialize our dataset.
if accelerator.is_main_process:
    print("Creating datasets...")
train_set, test_set = create_SkysatUnlabelled_dataset(
    os.path.join(config.dataset_path, "merged.csv"),  # merged.csv
    os.path.join(config.dataset_path, "merged"),
    preprocess_images,
    config.train_val_split,
)

train_dl = DataLoader(
    train_set, num_workers=2, batch_size=config.per_device_train_batch_size
)
test_dl = DataLoader(
    test_set, num_workers=2, batch_size=config.per_device_eval_batch_size
)

if accelerator.is_main_process:
    print("Train set batch size:", config.per_device_train_batch_size)
    print("Train set batch count:", len(train_dl))
    print("Test set batch size:", config.per_device_eval_batch_size)
    print("Test set batch count:", len(test_dl))

    print(process.memory_info().rss / 1024**3, "GB memory used")

# -----------------
# MODEL
# -----------------
if accelerator.is_main_process:
    print("Loading model:")
# extractor = AutoFeatureExtractor.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
model = Swinv2ForMaskedImageModeling.from_pretrained(pretrained_model_path)

if accelerator.is_main_process:
    print(process.memory_info().rss / 1024**3, "GB memory used")

# -----------------
# OPTIMIZER
# -----------------
optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=config.adam_lr,
    betas=config.adam_betas,
    eps=config.adam_eps,
    weight_decay=config.adam_weight_decay,
)

if config.scheduler == "linear":
    scheduler = LinearLR(
        optimizer,
        start_factor=config.start_factor,
        total_iters=config.lr_warmup_steps,
    )
elif config.scheduler == "cosine_warm":
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.max_train_steps,
        num_cycles=config.lr_num_cycles,
    )
elif config.scheduler == "cosine":
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.max_train_steps,
    )


# -----------------
# ACCELERATOR
# -----------------
model, optimizer, train_dl, scheduler = accelerator.prepare(
    model, optimizer, train_dl, scheduler
)


# -----------------
# EVAL METRICS
# -----------------
def get_eval_metric(gt_images, pred_images, avg=True):
    metric_list = ["mse", "pcc"]
    res_list = []

    samples_to_run = np.arange(1, len(gt_images)) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            res = get_similarity_metric(
                pred_images, gt_images, method="metrics-only", metric_name=m
            )
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))

    return res_list, metric_list


# -----------------
# TRAINING
# -----------------
best_val = 100.0
sidelen_patches = model_config.image_size // config.model_patch_size
step_count = 0

num_steps_per_epoch = math.ceil(len(train_dl) / config.gradient_accumulation_steps)
# Afterwards we recalculate our number of training epochs
num_train_epochs = math.ceil(config.max_train_steps / num_steps_per_epoch)

for epoch in range(num_train_epochs):
    if accelerator.is_main_process:
        print(f"Epoch {epoch + 1}/{num_train_epochs}")
    # TRAIN LOOP
    model.train()
    train_loss_list = []
    for batch_index, train_batch in enumerate(train_dl):
        with accelerator.accumulate(model):
            if accelerator.is_main_process:
                print("Step: ", batch_index, end="\r")
            # print(process.memory_info().rss / 1024 ** 3 , "GB memory used")
            optimizer.zero_grad()
            outputs = model(
                train_batch["pixel_values"], bool_masked_pos=train_batch["mask"]
            )
            loss = outputs.loss
            accelerator.backward(loss)
            train_loss_list.append(loss.item())  # do not track trace for this
            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            step_count += 1
            if step_count >= config.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        mean_loss = np.mean(train_loss_list)
        wandb.log({"train_loss": mean_loss}, commit=False)
        wandb.log({"epoch": epoch}, commit=False)

        # VALIDATION LOOP
        with torch.no_grad():
            model.eval()
            for batch_index, val_batch in enumerate(test_dl):
                # Our batch size is the entire test dataset, we should not have more than 1 batch
                if batch_index != 0:
                    break
                # Load to device
                val_batch["pixel_values"] = val_batch["pixel_values"].to(device)
                val_batch["mask"] = val_batch["mask"].to(device)
                # Log full validation images
                outputs = model(
                    val_batch["pixel_values"], bool_masked_pos=val_batch["mask"]
                )
                reconstructions = (
                    torch.clamp(
                        ((outputs.reconstruction + 1.0) / 2.0), min=0.0, max=1.0
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                originals = (
                    torch.clamp(
                        ((val_batch["pixel_values"] + 1.0) / 2.0), min=0.0, max=1.0
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                masks = val_batch["mask"].detach().cpu().numpy()
                # Create a figure showing original, masked and reconstructed images
                fig, axs = plt.subplots(originals.shape[0], 3, figsize=(10, 30))
                for i in range(originals.shape[0]):
                    axs[i, 0].imshow(
                        originals[i].transpose(1, 2, 0),
                        vmin=-1,
                        vmax=1,
                    )
                    axs[i, 0].axis("off")
                    # Reshape the array to 64x64
                    mask_2d = masks[i].reshape((sidelen_patches, sidelen_patches))
                    # Scale the array by a factor of 4 to get a 256x256 array
                    mask_scaled = mask_2d.repeat(
                        config.model_patch_size, axis=0
                    ).repeat(config.model_patch_size, axis=1)
                    axs[i, 1].imshow(
                        originals[i].transpose(1, 2, 0) * mask_scaled[:, :, np.newaxis],
                        vmin=-1,
                        vmax=1,
                    )
                    axs[i, 1].axis("off")
                    axs[i, 2].imshow(
                        reconstructions[i].transpose(1, 2, 0),
                        vmin=-1,
                        vmax=1,
                    )
                    axs[i, 2].axis("off")
                wandb.log({"reconstructions": wandb.Image(fig)}, commit=False)
                plt.close(fig)
                # Log evaluation metrics
                loss = outputs.loss
                metric, metric_names = get_eval_metric(
                    originals, reconstructions, avg=True
                )
                metric_dict = {f"val/{k}": v for k, v in zip(metric_names, metric)}
                wandb.log(metric_dict, commit=False)
                wandb.log({"val_loss": loss}, commit=False)

                # Save model if we are better than the best model so far
                if metric_dict["val/mse"] < best_val:
                    best_val = metric_dict["val/mse"]
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        os.path.join(config.output_path, "checkpoint_best"),
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )

        wandb.log({"epoch": epoch}, commit=True)

    accelerator.wait_for_everyone()
    scheduler.step()

accelerator.end_training()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    os.path.join(config.output_path, "checkpoint_final"),
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)
