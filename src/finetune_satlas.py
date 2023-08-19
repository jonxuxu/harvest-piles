import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
import torchvision
import torchvision.models.swin_transformer as swin
from torchmetrics import Accuracy, F1Score, AUROC, Precision, Recall

import wandb
import os
import numpy as np

from dataset import SkysatLabelled
from config import Config_Satlas

config = Config_Satlas()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed
torch.manual_seed(config.seed)
np.random.seed(config.seed)

# -----------------
# LOGGER
# -----------------
LOGGING = True
print("Init logger", LOGGING)
if LOGGING:
    run = wandb.init(
        project=config.wandb_project,
        group=config.wandb_group,
        config=vars(config),
    )

# -----------------
# DATASET
# -----------------
print("Loading datasets")

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        swin.Swin_V2_B_Weights.IMAGENET1K_V1.transforms(),
    ]
)

train_dataset = SkysatLabelled(
    os.path.join(config.dataset_path, "train.csv"),
    os.path.join(config.dataset_path, "merged"),
    transform,
)
test_dataset = SkysatLabelled(
    os.path.join(config.dataset_path, "test.csv"),
    os.path.join(config.dataset_path, "merged"),
    transform,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=config.val_batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

# -----------------
# MODEL
# -----------------
print("Setting model")

model = swin.swin_v2_b()
full_state_dict = torch.load(config.checkpoint_path)
swin_prefix = "backbone.backbone."
swin_state_dict = {
    k[len(swin_prefix) :]: v
    for k, v in full_state_dict.items()
    if k.startswith(swin_prefix)
}
model.load_state_dict(swin_state_dict)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Add a FC layer
num_features = model.head.out_features
num_classes = 1
model = torch.nn.Sequential(
    model, torch.nn.Linear(num_features, num_classes), torch.nn.Sigmoid()
)
# num_features = model.head.in_features
# model.head = torch.nn.Sequential(
#     torch.nn.Linear(num_features, num_classes), torch.nn.Sigmoid()
# )

# Load checkpoint
if config.load_trained:
    model.load_state_dict(torch.load(config.trained_path))

# -----------------
# CRITERION
# -----------------
if config.criterion == "classification":
    criterion = nn.BCEWithLogitsLoss()

accuracy = Accuracy(task="binary").to(device)
f1_score = F1Score(task="binary").to(device)
auroc = AUROC(task="binary").to(device)
precision = Precision(task="binary").to(device)
recall = Recall(task="binary").to(device)

# -----------------
# OPTIMIZER, SCHEDULER
# -----------------
if config.optimizer == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

if config.scheduler == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=config.lr_decay
    )
elif config.scheduler == "warmup_cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.max_epochs
    )
elif config.scheduler == "linear":
    scheduler = LinearLR(
        optimizer,
        start_factor=config.start_factor,
        total_iters=config.lr_warmup_steps,
    )

# -----------------
# ACCELERATOR
# -----------------
model = model.to(device)
criterion = criterion.to(device)


# -----------------
# TRAIN
# -----------------
def train(model, iterator, optimizer, criterion, scheduler, device):
    epoch_loss = 0

    model.train()
    for x, y, _ in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()
        if config.scheduler == "one_cycle_lr":
            scheduler.step()

        epoch_loss += loss.item()

    if config.scheduler != "one_cycle_lr":
        scheduler.step()

    epoch_loss /= len(iterator)

    return epoch_loss


def evaluate(model, iterator, criterion, device):
    model.eval()

    with torch.no_grad():
        x, y, _ = next(iter(iterator))
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        preds = (logits > 0.5).float()

        evals = {
            "acc": accuracy(preds, y),
            "f1": f1_score(preds, y),
            "auroc": auroc(preds, y),
            "precision": precision(preds, y),
            "recall": recall(preds, y),
        }

    return loss.item(), evals


print("Begin train")
print("Batch size:", config.batch_size)
print("Steps per epoch:", len(train_dataloader))

best_accuracy = 0

for epoch in range(config.max_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion, scheduler, device)
    valid_loss, val_metrics = evaluate(model, test_dataloader, criterion, device)
    if LOGGING:
        wandb.log({"train_loss": train_loss}, commit=False)
        wandb.log({"val": val_metrics}, commit=False)

    if val_metrics["acc"] > best_accuracy:
        best_accuracy = val_metrics["acc"]
        torch.save(model.state_dict(), config.output_path)

    print(f"Epoch: {epoch+1:02}")

    if LOGGING:
        wandb.log({"epoch": epoch}, commit=True)
