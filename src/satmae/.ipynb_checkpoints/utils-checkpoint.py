# utils
import copy
import math

import numpy as np
import torch
import os
import random
import torchvision
import warnings
from typing import Optional
from src.models import resnet18, resnet34, resnet50, resnext50_32x4d
from src.models_vit import vit_base_patch16,vit_large_patch16, ViTFinetune
import torch.nn.functional as F
import torchmetrics
# from configs import args
model_type = dict(
    resnet18=resnet18,
    resnet34=resnet34,
    resnet50=resnet50,
    resnext=resnext50_32x4d,
    vit=vit_base_patch16,
    vitL=vit_large_patch16,

)


def init_model(method, ckpt_path=None):
    '''
    :param method: str one of ['ckpt', 'imagenet', 'random']
    :param ckpt_path: str checkpoint path
    :return: tuple (ckpt_path:str , pretrained:bool)
    '''
    if method == 'ckpt':
        if ckpt_path:
            return ckpt_path, False
        else:
            raise ValueError('checkpoint path isnot provided')
    elif method == 'imagenet':
        return None, True
    else:
        return None, False


def get_model(model_name, in_channels, pretrained=False, ckpt_path=None):
    
    if 'satmae' in model_name:
        #vitL parameters
        model = ViTFinetune(
                img_size=224,
                patch_size=16,
                in_chans=3,
                num_classes=1,
                embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                drop_rate=0.1,
            )
   
    else:
        model_fn = model_type[model_name]
        model = model_fn(in_channels, pretrained)
    
    if ckpt_path:
        print(f'loading model from checkpoint{ckpt_path}')
        model = load_from_checkpoint(ckpt_path, model)
    return model




def load_from_checkpoint(path, model):
    
    # if the model path is "fmow_pretrain.pth"
    if 'fmow' in path:
        checkpoint=torch.load(path)
        checkpoint_model=checkpoint['model']
        state_dict = model.state_dict()

        loaded_dict = checkpoint_model
        model_dict = model.state_dict()
        del loaded_dict['mask_token']
        del loaded_dict['decoder_pos_embed']
      
        for key_model in model_dict.keys():
         
         if 'fc'  in key_model or 'head' in key_model :
               #ignore fc weight
              model_dict[key_model]= model_dict[key_model]
         else:
             model_dict[key_model]=loaded_dict[key_model]
          
        model.load_state_dict(model_dict)

    else:
        ckpt = torch.load(path)

        model.load_state_dict(ckpt)
    model.to('cuda')
    #model.eval()
    return model


