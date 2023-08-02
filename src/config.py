import os
import datetime

ROOT_PATH = "/atlas2/u/jonxuxu"


# Adding text labels to images in these datasets
class Config_Resnet:
    def __init__(self):
        # paths
        self.working_dir = os.path.join(ROOT_PATH, "harvest-piles")
        self.dataset_path = os.path.join(ROOT_PATH, "datasets")

        self.one_cycle_lr = 1e-3
        self.num_train_epochs = 15
