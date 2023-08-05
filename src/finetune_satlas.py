import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.models.swin_transformer as swin
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

# -----------------
# CRITERION
# -----------------
if config.criterion == "classification":
    criterion = nn.BCEWithLogitsLoss()


def binary_accuracy(y_true, y_prob):
    assert y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)


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


# -----------------
# ACCELERATOR
# -----------------
model = model.to(device)
criterion = criterion.to(device)


# -----------------
# TRAIN
# -----------------
print("Begin train")


def train(model, iterator, optimizer, criterion, scheduler, device):
    epoch_loss = 0
    epoch_accuracy = 0

    model.train()
    for x, y, _ in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)
        acc = binary_accuracy(y, y_pred)

        loss.backward()
        optimizer.step()
        if config.scheduler == "one_cycle_lr":
            scheduler.step()

        epoch_loss += loss.item()
        epoch_accuracy += acc

    if config.scheduler != "one_cycle_lr":
        scheduler.step()

    epoch_loss /= len(iterator)
    epoch_accuracy /= len(iterator)

    return epoch_loss, epoch_accuracy


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_accuracy = 0

    model.eval()

    with torch.no_grad():
        for x, y, _ in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = binary_accuracy(y, y_pred)

            epoch_loss += loss.item()
            epoch_accuracy += acc

    epoch_loss /= len(iterator)
    epoch_accuracy /= len(iterator)

    return epoch_loss, epoch_accuracy


best_valid_loss = float("inf")

for epoch in range(config.max_epochs):
    train_loss, train_acc = train(
        model, train_dataloader, optimizer, criterion, scheduler, device
    )
    valid_loss, valid_acc = evaluate(model, test_dataloader, criterion, device)
    if LOGGING:
        wandb.log({"train_loss": train_loss}, commit=False)
        wandb.log({"train_acc": train_acc}, commit=False)
        wandb.log({"valid_loss": valid_loss}, commit=False)
        wandb.log({"valid_acc": valid_acc}, commit=False)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), config.output_path)

    print(f"Epoch: {epoch+1:02}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:6.2f}%")
    print(f"\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:6.2f}%")

    if LOGGING:
        wandb.log({"epoch": epoch}, commit=True)
