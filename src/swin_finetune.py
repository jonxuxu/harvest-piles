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
from accelerate import Accelerator

from transformers import AutoModel
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

from transformers.utils import check_min_version

from config import Config_Swin_Finetune
from dataset import SkysatLabelled

""" Fine tuning our swin Transformer for binary classification of farm activity
"""

# -----------------
# CONFIG
# -----------------
config = Config_Swin_Finetune()
os.makedirs(config.output_path, exist_ok=True)

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
# MODEL
# -----------------
if accelerator.is_main_process:
    print("Loading model:")


class SwinClassifier(torch.nn.Module):
    def __init__(self, config, num_classes):
        super(SwinClassifier, self).__init__()

        self.swin_encoder = AutoModel.from_pretrained(config.pretrain_path)
        self.swin_encoder.requires_grad_(False)

        self.classifier = torch.nn.Linear(self.swin_encoder.num_features, num_classes)

    def forward(self, x):
        outputs = self.swin_encoder(x)
        pred = self.classifier(outputs[1])
        return pred


out_classes = 1
model = SwinClassifier(config, out_classes)
swin_config = model.swin_encoder.config

if accelerator.is_main_process:
    print(process.memory_info().rss / 1024**3, "GB memory used")


# -----------------
# DATASET
# -----------------
# transformations as done in original SimMIM paper
# source: https://github.com/microsoft/SimMIM/blob/main/data/data_simmim.py
transforms = Compose(
    [
        ToPILImage(),
        Resize((swin_config.image_size, swin_config.image_size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
        Normalize(
            mean=[0.412, 0.368, 0.326], std=[0.110, 0.097, 0.098]
        ),  # our dataset vals
    ]
)

# Initialize our dataset.
if accelerator.is_main_process:
    print("Creating datasets...")

train_dataset = SkysatLabelled(
    os.path.join(config.dataset_path, "train.csv"),
    os.path.join(config.dataset_path, "merged"),
    transforms,
)
test_dataset = SkysatLabelled(
    os.path.join(config.dataset_path, "test.csv"),
    os.path.join(config.dataset_path, "merged"),
    transforms,
)

train_dl = DataLoader(
    train_dataset,
    num_workers=2,
    batch_size=config.per_device_train_batch_size,
    pin_memory=True,
    shuffle=True,
)
test_dl = DataLoader(
    test_dataset,
    num_workers=2,
    batch_size=config.per_device_eval_batch_size,
    pin_memory=True,
    shuffle=True,
)

if accelerator.is_main_process:
    print("Train set batch size:", config.per_device_train_batch_size)
    print("Train set batch count:", len(train_dl))
    print("Test set batch size:", config.per_device_eval_batch_size)
    print("Test set batch count:", len(test_dl))

    print(process.memory_info().rss / 1024**3, "GB memory used")

# -----------------
# OPTIMIZER
# -----------------
criterion = torch.nn.BCEWithLogitsLoss()


def binary_accuracy(labels, logits):
    assert labels.size() == logits.size()
    predicted_probs = torch.sigmoid(logits)
    preds = (predicted_probs > 0.5).float()
    correct_predictions = (preds == labels).float()
    accuracy = correct_predictions.mean()
    return accuracy.item()


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
model, optimizer, criterion, train_dl, test_dl, scheduler = accelerator.prepare(
    model, optimizer, criterion, train_dl, test_dl, scheduler
)

# -----------------
# TRAINING
# -----------------
best_val = -1
step_count = 0

num_steps_per_epoch = math.ceil(len(train_dl) / config.gradient_accumulation_steps)
# Afterwards we recalculate our number of training epochs
num_train_epochs = math.ceil(config.max_train_steps / num_steps_per_epoch)

for epoch in range(num_train_epochs):
    if accelerator.is_main_process:
        print(f"Epoch {epoch + 1}/{num_train_epochs}")
    # TRAIN LOOP
    model.train()
    train_epoch_loss = 0
    train_epoch_acc = 0
    for batch_index, train_batch in enumerate(train_dl):
        with accelerator.accumulate(model):
            if accelerator.is_main_process:
                print("Step: ", batch_index, end="\r")
            # print(process.memory_info().rss / 1024 ** 3 , "GB memory used")
            optimizer.zero_grad()
            preds = model(train_batch[0])
            loss = criterion(preds, train_batch[1])
            acc = binary_accuracy(preds, train_batch[1])

            accelerator.backward(loss)
            train_epoch_loss += loss.item()  # do not track trace for this
            train_epoch_acc += acc

            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            step_count += 1
            if step_count >= config.max_train_steps:
                break
    train_epoch_loss /= len(train_dl)
    train_epoch_acc /= len(train_dl)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # VALIDATION LOOP
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()

            for batch_index, val_batch in enumerate(test_dl):
                # Our batch size is the entire test dataset, we should not have more than 1 batch
                if batch_index != 0:
                    break
                # Log full validation images
                preds = model(val_batch[0])

                # Log evaluation metrics
                loss = criterion(preds, val_batch[1])
                acc = binary_accuracy(preds, val_batch[1])
                val_epoch_loss += loss.item()
                val_epoch_acc += acc

                # Save model if we are better than the best model so far
                if acc > best_val:
                    best_val = acc
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(
                        unwrapped_model.state_dict(),
                        os.path.join(config.output_path, "model_best.pth"),
                    )
            val_epoch_loss /= len(test_dl)
            val_epoch_acc /= len(test_dl)

        wandb.log({"train/loss": train_epoch_loss}, commit=False)
        wandb.log({"train/acc": train_epoch_acc}, commit=False)
        wandb.log({"val/loss": val_epoch_loss}, commit=False)
        wandb.log({"val/acc": val_epoch_acc}, commit=False)

        wandb.log({"epoch": epoch}, commit=True)

    accelerator.wait_for_everyone()
    scheduler.step()

accelerator.end_training()
unwrapped_model = accelerator.unwrap_model(model)
torch.save(
    unwrapped_model.state_dict(), os.path.join(config.output_path, "model_final.pth")
)
