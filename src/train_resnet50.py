import os
import time
import torch
from torch.utils.data import DataLoader
import torchvision
import wandb
import numpy as np
from torchmetrics import Accuracy, F1Score, AUROC, Precision, Recall
from accelerate import Accelerator
from collections import defaultdict

from dataset import SkysatLabelled
from config import Config_Resnet

# -----------------
# CONFIG
# -----------------

config = Config_Resnet()

# Seed
torch.manual_seed(config.seed)
np.random.seed(config.seed)


accelerator = Accelerator(
    log_with="wandb",
    mixed_precision=config.mixed_precision,
)
device = accelerator.device

# Log on each process the small summary:
if accelerator.is_main_process:
    print(f"Training/evaluation parameters:")
    print(config.__dict__)

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

LOGGING = True
if LOGGING:
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
if accelerator.is_main_process:
    print("Loading datasets")
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
        # torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.412, 0.368, 0.326], std=[0.110, 0.097, 0.098]
        ),  # our dataset vals
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
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

# -----------------
# MODEL
# -----------------
if accelerator.is_main_process:
    print("Loading model")

model_weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2


class ResNet50(torch.nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(weights=model_weights)
        num_features = self.resnet50.fc.out_features
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_classes),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.resnet50(x)
        return self.fc(logits)


out_classes = 1
model = ResNet50(out_classes)

# -----------------
# OPTIMIZER, SCHEDULER
# -----------------
criterion = torch.nn.BCELoss()

accuracy = Accuracy(task="binary")
f1_score = F1Score(task="binary")
auroc = AUROC(task="binary")
precision = Precision(task="binary")
recall = Recall(task="binary")

FOUND_LR = config.lr  # For OneCycleLR
# FOUND_LR = 0.001 # For ExponentialLR


params = [
    {"params": model.resnet50.conv1.parameters(), "lr": FOUND_LR / 10},
    {"params": model.resnet50.bn1.parameters(), "lr": FOUND_LR / 10},
    {"params": model.resnet50.layer1.parameters(), "lr": FOUND_LR / 8},
    {"params": model.resnet50.layer2.parameters(), "lr": FOUND_LR / 6},
    {"params": model.resnet50.layer3.parameters(), "lr": FOUND_LR / 4},
    {"params": model.resnet50.layer4.parameters(), "lr": FOUND_LR / 2},
    {"params": model.resnet50.fc.parameters()},
]
if config.optimizer == "adam":
    optimizer = torch.optim.Adam(params, lr=config.lr)
elif config.optimizer == "madgrad":
    from optim.madgrad import MADGRAD

    optimizer = MADGRAD(params, lr=config.lr)

STEPS_PER_EPOCH = len(train_dataloader)
TOTAL_STEPS = config.num_train_epochs * STEPS_PER_EPOCH
MAX_LRS = [p["lr"] for p in optimizer.param_groups]

if config.scheduler == "one_cycle_lr":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LRS, total_steps=TOTAL_STEPS
    )

# -----------------
# ACCELERATOR
# -----------------
model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, test_dataloader, scheduler
)
accuracy, f1_score, auroc, precision, recall = accelerator.prepare(
    accuracy, f1_score, auroc, precision, recall
)

# -----------------
# TRAIN
# -----------------
if accelerator.is_main_process:
    print("Begin train")


def train(model, iterator, optimizer, criterion, scheduler):
    epoch_loss = 0
    sum_metrics = defaultdict(float)

    model.train()
    for x, y, _ in iterator:
        optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)
        evals = {
            "acc": accuracy(pred, y),
            "f1": f1_score(pred, y),
            "auroc": auroc(pred, y),
            "precision": precision(pred, y),
            "recall": recall(pred, y),
        }
        for k, v in evals.items():
            sum_metrics[k] += v

        accelerator.backward(loss)
        optimizer.step()
        if config.scheduler == "one_cycle_lr":
            scheduler.step()

        epoch_loss += loss.item()

    if config.scheduler != "one_cycle_lr":
        scheduler.step()

    epoch_loss /= len(iterator)
    avg_metrics = {k: v / len(iterator) for k, v in sum_metrics.items()}

    return epoch_loss, avg_metrics


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    sum_metrics = defaultdict(float)

    model.eval()

    with torch.no_grad():
        for x, y, _ in iterator:
            pred = model(x)
            loss = criterion(pred, y)
            evals = {
                "acc": accuracy(pred, y),
                "f1": f1_score(pred, y),
                "auroc": auroc(pred, y),
                "precision": precision(pred, y),
                "recall": recall(pred, y),
            }
            for k, v in evals.items():
                sum_metrics[k] += v

            epoch_loss += loss.item()

    epoch_loss /= len(iterator)
    avg_metrics = {k: v / len(iterator) for k, v in sum_metrics.items()}

    return epoch_loss, avg_metrics


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_val = 0

if LOGGING:
    print("Batch size:", config.batch_size)
    print("Batches per epoch:", len(train_dataloader))
for epoch in range(config.num_train_epochs):
    start_time = time.time()

    train_loss, train_metrics = train(
        model, train_dataloader, optimizer, criterion, scheduler
    )
    valid_loss, val_metrics = evaluate(model, test_dataloader, criterion)
    if LOGGING:
        wandb.log({"train.loss": train_loss}, commit=False)
        wandb.log({"train": train_metrics}, commit=False)
        wandb.log({"val.loss": valid_loss}, commit=False)
        wandb.log({"val": val_metrics}, commit=False)

    if val_metrics["acc"] > best_val:
        best_val = val_metrics["acc"]
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(
            model.state_dict(), "/atlas2/u/jonxuxu/harvest-piles/results/resnet.pt"
        )

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if accelerator.is_main_process:
        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_metrics['acc'] * 100:6.2f}%"
        )
        print(
            f"\tValid Loss: {valid_loss:.3f} | Valid Acc: {val_metrics['acc'] * 100:6.2f}%"
        )

    if LOGGING:
        wandb.log({"epoch": epoch}, commit=True)
