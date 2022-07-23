# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_vit(ckpt):
    # 已看過，轉換用的函數

    # 構建一個最終要回傳的空間
    new_ckpt = OrderedDict()

    # 遍歷預訓練權重裏面的所有資訊
    for k, v in ckpt.items():
        # 簡單來說就是將一些相同層結構只是名稱不同的地方更改成MMSegmentation的名稱
        if k.startswith('head'):
            # head的部分就跳過
            continue
        if k.startswith('norm'):
            # 如果是norm就換成ln1
            new_k = k.replace('norm.', 'ln1.')
        elif k.startswith('patch_embed'):
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        elif k.startswith('blocks'):
            if 'norm' in k:
                new_k = k.replace('norm', 'ln')
            elif 'mlp.fc1' in k:
                new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                new_k = k.replace('mlp.fc2', 'ffn.layers.1')
            elif 'attn.qkv' in k:
                new_k = k.replace('attn.qkv.', 'attn.attn.in_proj_')
            elif 'attn.proj' in k:
                new_k = k.replace('attn.proj', 'attn.attn.out_proj')
            else:
                new_k = k
            new_k = new_k.replace('blocks.', 'layers.')
        else:
            new_k = k
        new_ckpt[new_k] = v

    return new_ckpt


def main():
    # 已看過，用來將timm的預訓練權重轉成MMSegmentation的型態
    # timm可以看成是torchvision的擴充版，裡面很多的模型以及預訓練權重同時準確度都還很高
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMSegmentation style.')
    # 預訓練權重檔案位置
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    # 轉換後要將權重放到的路徑，這裡必須要是絕對路徑
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    # 透過指定的預訓練權重位置讀取資料
    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint，如果是來自timm的預訓練權重就將state_dict資料拿出來，這會是我們要的資料
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint，如果是來自deit的預訓練權重就將model提取出來
        state_dict = checkpoint['model']
    else:
        # 其他就直接附值
        state_dict = checkpoint
    # 透過convert_vit進行轉換
    weight = convert_vit(state_dict)
    # 如果保存資料夾不存在就會新建一個
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    # 將資料進行保存
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
