import os
import time
import torch
from torch.utils.data import DataLoader
import torchvision
import wandb

from dataset import SkysatLabelled
from config import Config_Resnet

config = Config_Resnet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# LOGGER
# -----------------
run = wandb.init(
    # Set the project where this run will be logged
    project=config.wandb_project,
    group=config.wandb_group,
    # Track hyperparameters and run metadata
    config=vars(config),
)


# -----------------
# DATASET
# -----------------
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

train_dataloader = Dataloader(
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
print("Setting model")

model_weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2


class ResNet50(torch.nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(weights=model_weights)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_classes), torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet50(x)


out_classes = 1
model = ResNet50(out_classes)

# -----------------
# OPTIMIZER, SCHEDULER
# -----------------
criterion = torch.nn.BCELoss()

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
optimizer = torch.optim.Adam(params, lr=config.lr)

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
model = model.to(device)
criterion = criterion.to(device)

# -----------------
# TRAIN
# -----------------
print("Begin train")


def binary_accuracy(y_true, y_prob):
    assert y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)


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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float("inf")

for epoch in range(config.num_train_epochs):
    start_time = time.time()

    train_loss, train_acc = train(
        model, train_dataloader, optimizer, criterion, scheduler, device
    )
    wandb.log({"train_loss": train_loss}, commit=False)
    wandb.log({"train_acc": train_acc}, commit=False)

    valid_loss, valid_acc = evaluate(model, test_dataloader, criterion, device)
    wandb.log({"valid_loss": valid_loss}, commit=False)
    wandb.log({"valid_acc": valid_acc}, commit=False)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(
            model.state_dict(), "/atlas2/u/jonxuxu/harvest-piles/results/resnet.pt"
        )

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:6.2f}%")
    print(f"\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:6.2f}%")

    wandb.log({"epoch": epoch}, commit=True)
