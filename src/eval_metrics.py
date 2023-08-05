from einops import rearrange
import numpy as np


def smaller_the_better(gt, comp):
    return gt < comp


def larger_the_better(gt, comp):
    return gt > comp


def mse_metric(img1, img2):
    return (np.square(img1 - img2)).mean()


def pcc_metric(img1, img2):
    return np.corrcoef(img1.reshape(-1), img2.reshape(-1))[0, 1]


def metrics_only(pred_imgs, gt_imgs, metric, *args, **kwargs):
    return metric(pred_imgs, gt_imgs)


def get_similarity_metric(img1, img2, metric_name="mse", **kwargs):
    # img1: n, w, h, 3
    # img2: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    if img1.shape[-1] != 3:
        img1 = rearrange(img1, "n c w h -> n w h c")
    if img2.shape[-1] != 3:
        img2 = rearrange(img2, "n c w h -> n w h c")

    if metric_name == "mse":
        metric_func = mse_metric
        decision_func = smaller_the_better
    elif metric_name == "pcc":
        metric_func = pcc_metric
        decision_func = larger_the_better

    else:
        raise NotImplementedError

    return metrics_only(img1, img2, metric_func, decision_func, **kwargs)
