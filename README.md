# HarvestNet: A Dataset for Detecting Smallholder Farming Activity Using Harvest Piles and Remote Sensing

![Pile Examples](pile-examples.png)
Examples of harvest piles circled in red

## HarvestNet 
**HarvestNet** is a dataset for tracking farm activity by detecting harvest piles. This document introduces the procedures required for replicating the results in our paper.

## Abstract
Small farms contribute to a large share of the productive land in developing countries. In regions such as sub-Saharan Africa, where 80% of farms are small (under 2 ha in size), the task of mapping smallholder cropland is an important part of tracking sustainability measures such as crop productivity. However, the visually diverse and nuanced appearance of small farms has limited the effectiveness of traditional approaches to cropland mapping. Here we introduce a new approach based on the detection of harvest piles characteristic of many smallholder systems throughout the world. We present HarvestNet, a dataset for mapping the presence of farms in the Ethiopian regions of Tigray and Amhara during 2020-2023, collected using expert knowledge and satellite images, totaling 7k hand-labeled images and 2k ground collected labels. We also benchmark a set of baselines including SOTA models in remote sensing with our best models having around 80% classification performance on hand labelled data and 90%, 98% accuracy on ground truth data for Tigray, Amhara respectively. We also perform a visual comparison with a widely used pre-existing coverage map and show that our model detects an extra 56,621 hectares of cropland in Tigray. We conclude that remote sensing of harvest piles can contribute to more timely and accurate cropland assessments in food insecure regions.

## Overview
Our dataset consists of 7k labelled square SkySat images of size 512x512 pixels at a resolution of 0.5m per pixel. Each of these labelled images also correspond to a PlanetScope image of size 56x56 pixels at a resolution of 4.77m per pixel to cover the same geographic area of 256x256m. The labels are stored as `train.csv` and `test.csv`. Each row in the labelled dataset contains:

| Field | Description |
| ------------- | ------------- |
| filename | Name of the corresponding SkySat and PlanetScope image |
| lat_1 | Latitude of top left corner of area |
| lon_1 | Longitude of top left corner of area |
| lat_2 | Latitude of bottom right corner of area |
| lon_2 | Longitude of bottom right corner of area|
| activity | Label for whether the image contains harvest pile activity |
| altitude | Alttidue of the center of the image, in meters |
| lat_mean | Mean of lat_1 and lat_2 |
| lon_mean | Mean of lon_1 and lon_2 |
| year | Year of image capture |
| month | Month of image capture |
| day | Day of image capture |
| group | Contiguous overlapping group the area belongs to. If no overlap, assign group = -1 |


This dataset also includes ~150k unlabelled images SkySat images. They are of the same dimension with similar label format as our labelled dataset, without the `group` and `activity` fields defined. The labelled and unlabelled dataset are both included in 

The **datasets** folder and **weights** folder are not included in this repository. 
Please download them from TODO [FigShare](https://figshare.com/s/df347b379d0e2e01f30c) and put them in the root directory of this repository as shown below.

File path | Description
```

/datasets
┣ 📂 skysat_images
┃   ┗ 📜 0.tif
┃   ┗ ...
┃   ┗ 📜 xx.tif
┣ 📂 planetscope_images
┃   ┗ 📜 0.png
┃   ┗ ...
┃   ┗ 📜 xx.png

/weights
┣ 📂 swin_finetune
┣ 📂 swin_pretrain
┣ 📂 satmae_finetune
┗ 📜 resnet.pt
┗ 📜 satlas.pth

/src
┣ 📂 optim                      (custom optimizers)
┣ 📂 preprocessing              (helper scripts for creating dataset)
┣ 📂 scripts                    (helper scripts for running jobs on HPC)

┗ 📜 train.csv                  (labels for training set)
┗ 📜 test.csv                   (labels for test set)
┗ 📜 labels_all.csv             (labels for entire 150k dataset)

┗ 📜 finetune_satlas.py         (main script for fine-tuning Satlas classifier)
┗ 📜 swin_pretrain.py           (main script for pretraining Swin V2 MAE)
┗ 📜 swin_finetune.py           (main script for finetuning Swin V2 classifier)
┗ 📜 train_resnet.py            (main script for finetuning Resnet50 classifier)

┗ 📜 config.py                  (configurations for training scripts)
┗ 📜 dataset.py                 (functions for loading datasets)
┗ 📜 eval_metrics.py            (functions for evaluation metrics)

/notebooks
┗ 📜 Dataset_Explorer.ipynb     (printing grid of positive images, plot histograms of dataset distribution)
┗ 📜 Dataset_Maker.ipynb        (creating csv containing image labels from .tif images)
┗ 📜 Dataset_Split.ipynb        (create dataset for MTurks, applying expert labels, overlap partitioning algorithm, train test split scripts)
┗ 📜 Image_Load.ipynb           (remove corrupted images from dataset)
┗ 📜 Labelling.ipynb            (used by experts to label images)
┗ 📜 Migration.ipynb            (combine disjoint labels to one dataset)
┗ 📜 PlanetScope_Download.ipynb (download images from PlanetScope)
┗ 📜 SkySat_Clip_Bbox.ipynb     (create basic csv file for images in folder)
┗ 📜 SkySat_Clip.ipynb          (divide SkySat captures into 512x512 px images)
┗ 📜 SkySat_Download.ipynb      (download images from SkySat)
```

## Environment setup
Create and activate conda environment named ```harvest``` from our ```env.yaml```
```sh
conda env create -f env.yaml
conda activate harvest
```

## Download data
Due to size limit and license issues, the original SkySat images will need to be downloaded from the [Planet Explorer](https://www.planet.com/explorer/). The pre-processing scripts are also included in this repo.

1. SkySat_Download.ipynb: Notebook to download specified SkySat assets. Please refer to the Planet SDK for Python [repo](https://github.com/planetlabs/planet-client-python/tree/main) to set up your Planet account.
2. SkySat_Clip.ipynb: Notebook to clip given SkySat Collects into 512x512 px images and delete images that are partially empty.
3. SkySat_Clip_Bbox.ipynb: Notebook to extract bounding box coordinates of each SkySat clipped image to be used to download PlanetScope images.
4. PlanetScope_Download.ipynb: Notebook to download PlanetScope monthly basemaps using Google Earth Engine. Please refer to this NICFI access [page](https://developers.planet.com/docs/integrations/gee/nicfi/) to setup your Google Earth Engine account to gain access to collection of interest.
