import os
import datetime

ROOT_PATH = "/atlas2/u/jonxuxu"


# Adding text labels to images in these datasets
class Config_Resnet:
    def __init__(self):
        # paths
        self.working_dir = os.path.join(ROOT_PATH, "harvest-piles")
        self.dataset_path = os.path.join(ROOT_PATH, "datasets")

        self.wandb_project = "harvest-piles"
        self.wandb_group = "resnet50"

        self.scheduler = "one_cycle_lr"
        if self.scheduler == "one_cycle_lr":
            self.lr = 1e-3

        self.optimizer = "madgrad"

        self.num_train_epochs = 20

        self.batch_size = 32


class Config_Satlas:
    def __init__(self):
        # paths
        self.working_dir = os.path.join(ROOT_PATH, "harvest-piles")
        self.dataset_path = os.path.join(ROOT_PATH, "datasets")
        self.checkpoint_path = os.path.join(
            ROOT_PATH, "harvest-piles/weights/satlas-model-v1-highres.pth"
        )
        self.output_path = "/atlas2/u/jonxuxu/harvest-piles/results/satlas.pth"

        self.wandb_project = "harvest-piles"
        self.wandb_group = "satlas"

        self.train_split = 0.8

        self.criterion = "classification"

        self.optimizer = "adam"
        if self.optimizer == "adam":
            self.lr = 3e-4
            self.weight_decay = 0.1

        self.scheduler = "step"
        if self.scheduler == "step":
            self.lr_decay = 0.97
            self.patience = 20

        self.batch_size = 32
        self.val_batch_size = 8
        self.max_epochs = 100
