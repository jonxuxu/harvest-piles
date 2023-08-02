import os
import time
import torch
import torchvision

from dataset import Skysat
from config import Config_Resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# DATASET
# -----------------
print("Loading datasets")
train_dataset = Skysat(os.path.join(config.dataset_path, "train.csv"))
test_dataset = Skysat(os.path.join(config.dataset_path, "test.csv"))

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)

# -----------------
# MODEL
# -----------------

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

FOUND_LR = config.one_cycle_lr  # For OneCycleLR
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
optimizer = torch.optim.Adam(params, lr=config.one_cycle_lr)

STEPS_PER_EPOCH = len(train_dataloader)
TOTAL_STEPS = config.num_train_epochs * STEPS_PER_EPOCH
MAX_LRS = [p["lr"] for p in optimizer.param_groups]
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
def get_accuracy(y_true, y_prob):
    assert y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    # count = (y_true == y_prob).sum().item()
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
        acc = get_accuracy(y, y_pred)

        loss.backward()
        optimizer.step()
        scheduler.step()  # have scheduler here for other OneCycle scheduler

        epoch_loss += loss.item()
        epoch_accuracy += acc

    # scheduler.step() # have scheduler here for other LR schedulers

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

            acc = get_accuracy(y, y_pred)

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
    valid_loss, valid_acc = evaluate(model, val_dataloader, criterion, device)

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
