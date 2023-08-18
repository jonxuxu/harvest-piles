from pathlib import Path
# from mosaiks import config as c
from featurization import featurize, featurize_and_save

NUM_FEATS = 128
################################################################################################
# for 6km images
# FEATURIZATION
img_res = "250m"

config_features = {
    "random": {
        "patch_size": 3,
        "seed": 25,
        "type": "random_features",
        "num_filters": int(NUM_FEATS/2), # 4096,
        "pool_size": 256,
        "pool_stride": 256,
        "bias": 0.0,
        "filter_scale": 1e-3,
        "patch_distribution": "empirical",
    },
    "pretrained": {"model_type": "resnet50", "batch_size": 128},
}

# 6km images
images_dir = "/atlas/u/amna/mosaiks/imgs_v2"
################################################################################################

# # for 640m images

# # FEATURIZATION
# img_res = "640m"
# config_features = {
#     "random": {
#         "patch_size": 3,
#         "seed": 0,
#         "type": "random_features",
#         "num_filters":  int(NUM_FEATS/2),
#         "pool_size": 64,
#         "pool_stride": 64,
#         "bias": 0.0,
#         "filter_scale": 1e-3,
#         "patch_distribution": "empirical",
#     },
#     "pretrained": {"model_type": "resnet152", "batch_size": 128},
# }


# # 640m images
# images_dir = "/network/scratch/s/sara.ebrahim-elkafrawy/ecosystem_project/satellite_imgs/all_640m"
################################################################################################


feats_dir = "/atlas/u/amna/mosaiks/output_mosaiks"
image_folder = Path(images_dir) 

# remember to merge TRAIN and VAL
save_file_dir = Path(feats_dir) / f"{img_res}_{NUM_FEATS}_feats"
out_file_path = save_file_dir / f"{image_folder.name}_{NUM_FEATS}_feats.pkl"

assert save_file_dir.is_dir(), f"Output folder doesn't exist: {save_file_dir}"

assert (
    image_folder.is_dir()
), f"You have not downloaded images to {image_folder}"

featurize_and_save(image_folder, out_file_path, config_features) #c)