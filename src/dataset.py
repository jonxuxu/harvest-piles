import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import cv2
import torchvision


class SkysatLabelled(Dataset):
    def __init__(self, csv_file, image_dir, transform):
        df = pd.read_csv(csv_file, usecols=["filename", "activity"])
        df.reset_index(inplace=True, drop=True)
        self.x = df["filename"]
        self.y = torch.tensor(df["activity"]).unsqueeze(1).float()
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        path = os.path.join(self.image_dir, self.x[index])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (self.transform(image), self.y[index], self.x[index])


class SkysatUnlabelled(Dataset):
    def __init__(self, filenames, image_dir, transform):
        self.x = filenames
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        path = os.path.join(self.image_dir, self.x[index])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image)


def create_SkysatUnlabelled_dataset(csv_file, image_dir, transform, train_split):
    df = pd.read_csv(csv_file, usecols=["filename"])
    df.reset_index(inplace=True, drop=True)
    examples = df["filename"]
    train_size = int(len(examples) * train_split)
    train_examples = examples[:train_size]
    test_examples = examples[train_size:]

    return SkysatUnlabelled(train_examples, image_dir, transform), SkysatUnlabelled(
        test_examples, image_dir, transform
    )
