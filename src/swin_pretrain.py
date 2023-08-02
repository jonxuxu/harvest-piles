import os
import math
import wandb
import psutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
import matplotlib.pyplot as plt
from accelerate import Accelerator

from transformers import (
    Swinv2Config,
)
from transformers.utils import check_min_version
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup

# from datasets import load_dataset

from config import Config_Resnet
from dataset import create_skysat_dataset
from eval_metrics import get_similarity_metric

from mae.swin_mae import Swinv2ForMaskedImageModeling

""" Pre-training a 🤗 Transformers model for simple masked image modeling (SimMIM).
Any model supported by the AutoModelForMaskedImageModeling API can be used.
"""


def main():
    # -----------------
    # CONFIG
    # -----------------
    config = Config_MAE_Pretrain()
    os.makedirs(config.output_path, exist_ok=True)

    model_config = Swinv2Config(num_channels=1)

    with open(os.path.join(config.output_path, "README.md"), "w+") as f:
        print(config.__dict__, file=f)

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version("4.30.0.dev0")

    # Device
    print("Using GPU: {}".format(torch.cuda.is_available()))
    print("GPU count: {}".format(torch.cuda.device_count()))

    # Accelerate
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        mixed_precision=config.mixed_precision,
    )

    # Log on each process the small summary:
    accelerator.print(f"Training/evaluation parameters:")
    accelerator.print(config.__dict__)

    # -----------------
    # LOGGER
    # -----------------
    process = psutil.Process()
    accelerator.print(process.memory_info().rss / 1024**3, "GB memory used")

    accelerator.init_trackers(
        config.project_name,
        config=config,
        init_kwargs={
            "wandb": {
                "group": config.group_name,
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
                raise ValueError(
                    "Mask patch size must be divisible by model patch size"
                )

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
            Resize((model_config.image_size, model_config.image_size)),  # 224
            ToTensor(),
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
    accelerator.print("Creating datasets...")
    if config.dataset == "HCP_2d":
        train_set, test_set = create_HCP2D_dataset(
            config.hcp2d_path, preprocess_images, config.train_val_split
        )
    elif config.dataset == "NSD_2d":
        train_set, test_set = create_NSD2D_dataset(
            config.nsd2d_path, preprocess_images, config.train_val_split
        )
        # dataset = load_dataset("clane9/NSD-Flat")

    train_dl = DataLoader(
        train_set,
        batch_size=config.per_device_train_batch_size,
        pin_memory=True,
        num_workers=4,
    )
    test_dl = DataLoader(
        test_set, batch_size=config.per_device_eval_batch_size, pin_memory=True
    )

    accelerator.print("Train set batch size:", config.per_device_train_batch_size)
    accelerator.print("Train set batch count:", len(train_dl))
    accelerator.print("Test set batch size:", config.per_device_eval_batch_size)
    accelerator.print("Test set batch count:", len(test_dl))

    accelerator.print(process.memory_info().rss / 1024**3, "GB memory used")

    # -----------------
    # MODEL
    # -----------------
    accelerator.print("Loading model:")
    # extractor = AutoFeatureExtractor.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
    model = Swinv2ForMaskedImageModeling(model_config)
    accelerator.print(process.memory_info().rss / 1024**3, "GB memory used")

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
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.max_train_steps,
    )

    # -----------------
    # ACCELERATOR
    # -----------------
    model, optimizer, train_dl, test_dl, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dl, test_dl, lr_scheduler
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

    num_steps_per_epoch = math.ceil(
        len(train_dl)
        / config.gradient_accumulation_steps
        / accelerator.num_processes
        / config.per_device_train_batch_size
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(config.max_train_steps / num_steps_per_epoch)
    step_count = 0

    for epoch in range(num_train_epochs):
        accelerator.print(f"Epoch {epoch + 1}/{num_train_epochs}")
        # TRAIN LOOP
        model.train()
        train_loss_list = []
        step_count = 0
        for batch_index, train_batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                accelerator.print("Step: ", batch_index, end="\r")
                # print(process.memory_info().rss / 1024 ** 3 , "GB memory used")
                optimizer.zero_grad()
                outputs = model(
                    train_batch["pixel_values"], bool_masked_pos=train_batch["mask"]
                )
                loss = outputs.loss
                accelerator.backward(loss)
                train_loss_list.append(loss.item())  # do not track trace for this
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), config.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
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
                    # Log full validation images
                    if config.dataset == "HCP_2d":
                        dset_min = -10  # based on our dataset
                        dset_max = 10
                    elif config.dataset == "NSD_2d":
                        dset_min = -87  # based on our dataset
                        dset_max = 107
                    outputs = model(
                        val_batch["pixel_values"], bool_masked_pos=val_batch["mask"]
                    )
                    reconstructions = outputs.reconstruction.detach().cpu().numpy()
                    originals = val_batch["pixel_values"].detach().cpu().numpy()
                    # reconstructions = (
                    #     (
                    #         (outputs.reconstruction - dset_min)
                    #         * (1.0 / (dset_max - dset_min))
                    #     )
                    #     .detach()
                    #     .cpu()
                    #     .numpy()
                    # )
                    # originals = (
                    #     (
                    #         (val_batch["pixel_values"] - dset_min)
                    #         * (1.0 / (dset_max - dset_min))
                    #     )
                    #     .detach()
                    #     .cpu()
                    #     .numpy()
                    # )
                    masks = val_batch["mask"].detach().cpu().numpy()

                    # Create a figure showing original, masked and reconstructed images
                    # reconstructions_display = np.copy(reconstructions)
                    # originals_display = np.copy(originals)
                    fig, axs = plt.subplots(originals.shape[0], 3, figsize=(10, 30))
                    for i in range(originals.shape[0]):
                        # set background color to white
                        # if config.dataset == "HCP_2d":
                        #     bg_value = 0.5
                        # elif config.dataset == "NSD_2d":
                        #     bg_value = 0.44845361
                        # bg_index = np.isclose(
                        #     reconstructions_display, bg_value, rtol=0, atol=1e-6
                        # )
                        # reconstructions_display[bg_index] = 1
                        # bg_index = np.isclose(
                        #     originals_display, bg_value, rtol=0, atol=1e-6
                        # )
                        # originals_display[bg_index] = 1
                        axs[i, 0].imshow(
                            originals[i].transpose(1, 2, 0),
                            cmap="gray",
                            vmin=dset_min,
                            vmax=dset_max,
                        )
                        axs[i, 0].axis("off")
                        # Reshape the array to 64x64
                        mask_2d = masks[i].reshape((sidelen_patches, sidelen_patches))
                        # Scale the array by a factor of 4 to get a 256x256 array
                        mask_scaled = mask_2d.repeat(
                            config.model_patch_size, axis=0
                        ).repeat(config.model_patch_size, axis=1)
                        axs[i, 1].imshow(
                            originals[i].transpose(1, 2, 0)
                            * mask_scaled[:, :, np.newaxis],
                            cmap="gray",
                            vmin=dset_min,
                            vmax=dset_max,
                        )
                        axs[i, 1].axis("off")
                        axs[i, 2].imshow(
                            reconstructions[i].transpose(1, 2, 0),
                            cmap="gray",
                            vmin=0.0,
                            vmax=1.0,
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

                    # Save model if we are better than the best model so far
                    if metric_dict["val/mse"] < best_val:
                        best_val = metric_dict["val/mse"]
                        unwrappedModel = accelerator.unwrap_model(model)
                        unwrappedModel.save_pretrained(
                            os.path.join(config.output_path, "checkpoint_best")
                        )

            wandb.log({"epoch": epoch}, commit=True)

    accelerator.end_training()
    unwrappedModel = accelerator.unwrap_model(model)
    unwrappedModel.save_pretrained(os.path.join(config.output_path, "checkpoint_final"))


if __name__ == "__main__":
    main()
