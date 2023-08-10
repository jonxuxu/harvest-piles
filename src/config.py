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

        self.seed = 2023

        self.scheduler = "one_cycle_lr"
        if self.scheduler == "one_cycle_lr":
            self.lr = 1e-3

        self.optimizer = "adam"

        self.num_train_epochs = 20

        self.batch_size = 32


class Config_Satlas:
    def __init__(self):
        # paths
        self.working_dir = os.path.join(ROOT_PATH, "harvest-piles")
        self.dataset_path = os.path.join(ROOT_PATH, "datasets")
        self.checkpoint_path = os.path.join(
            self.working_dir, "weights/satlas-model-v1-highres.pth"
        )
        self.trained_path = os.path.join(self.working_dir, "weights/satlas-trained.pth")
        self.output_path = "/atlas2/u/jonxuxu/harvest-piles/results/satlas.pth"

        self.wandb_project = "harvest-piles"
        self.wandb_group = "satlas"

        self.seed = 2023
        self.train_split = 0.8
        self.criterion = "classification"
        self.load_trained = True

        self.optimizer = "adam"
        if self.optimizer == "adam":
            self.lr = 3e-5  # late stage lr
            self.weight_decay = 0.1

        self.scheduler = "warmup_cosine"
        if self.scheduler == "step":
            self.lr_decay = 0.97
            self.patience = 20
        elif self.scheduler == "linear":
            self.start_factor = 0.33
            self.lr_warmup_steps = 5

        self.batch_size = 50
        self.val_batch_size = 460
        self.max_epochs = 100


class Swin_Pretrain:
    def __init__(self):
        # paths
        self.working_dir = os.path.join(ROOT_PATH, "harvest-piles")
        self.dataset_path = os.path.join(ROOT_PATH, "datasets")
        self.output_path = os.path.join(
            "/atlas2/u/jonxuxu/harvest-piles/results/swin_pretrain",
            "%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
        )

        self.wandb_project = "harvest-piles"
        self.wandb_group = "swin_pretrain"

        self.seed = 2023

        # adam optimizer
        self.adam_lr = 1.5e-4
        self.adam_betas = (0.9, 0.999)
        self.adam_eps = 1e-08
        self.adam_weight_decay = 0.05

        # lr scheduler
        self.scheduler = "linear"
        if self.scheduler == "linear":
            self.start_factor = 0.33
            self.lr_warmup_steps = 5
        elif self.scheduler == "cosine_warmup":
            self.lr_num_cycles = 100

        # model train args
        self.gradient_accumulation_steps = 1
        self.mixed_precision = "fp16"

        self.per_device_train_batch_size = 70
        self.per_device_eval_batch_size = 5

        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0

        # data
        self.max_train_steps = 100000
        self.train_val_split = 0.95
        self.mask_patch_size = 32
        self.mask_ratio = 0.6
        self.model_patch_size = 4


class Config_Swin_Finetune:
    def __init__(self):
        # paths
        self.working_dir = os.path.join(ROOT_PATH, "harvest-piles")
        self.dataset_path = os.path.join(ROOT_PATH, "datasets")
        self.output_path = os.path.join(
            "/atlas2/u/jonxuxu/harvest-piles/results/swin_finetune",
            "%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
        )
        self.pretrain_path = os.path.join(
            ROOT_PATH, "harvest-piles/weights/swin_mae_pretrain"
        )
        self.trained_path = os.path.join(
            ROOT_PATH, "harvest-piles/weights/swin_mae_finetune/model_best.pth"
        )

        self.wandb_project = "harvest-piles"
        self.wandb_group = "swin_finetune"

        self.seed = 2023
        self.load_trained = True

        # adam optimizer
        self.adam_lr = 3e-4  # 1e-3 for train
        self.adam_betas = (0.9, 0.999)
        self.adam_eps = 1e-08
        self.adam_weight_decay = 0.05

        # lr scheduler
        self.scheduler = "linear"
        if self.scheduler == "linear":
            self.start_factor = 0.33
            self.lr_warmup_steps = 5
        elif self.scheduler == "cosine_warmup":
            self.lr_num_cycles = 100

        # model train args
        self.gradient_accumulation_steps = 1
        self.mixed_precision = "fp16"

        self.per_device_train_batch_size = 50
        self.per_device_eval_batch_size = 500

        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0

        # data
        self.max_train_steps = 5000
        self.train_val_split = 0.80
