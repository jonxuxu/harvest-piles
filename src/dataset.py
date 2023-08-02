import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import cv2
import torchvision

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
        # torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #imagenet vals
        torchvision.transforms.Normalize(
            mean=[0.412, 0.368, 0.326], std=[0.110, 0.097, 0.098]
        ),  # our dataset vals
    ]
)


class Skysat(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path, usecols=["filename", "activity"])
        df.reset_index(inplace=True, drop=True)
        self.x = df["filename"]
        self.y = torch.tensor(df["activity"]).unsqueeze(1).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        path = os.path.join(DATASET_PATH, "merged", self.x[index])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (transform(image), self.y[index], self.x[index])
