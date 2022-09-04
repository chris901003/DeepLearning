# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys

import mmcv
from mmcv import DictAction
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmgen.apis import init_model, sample_unconditional_model  # isort:skip  # noqa
# yapf: enable


def parse_args():
    # 生成圖像參數
    parser = argparse.ArgumentParser(description='Generation demo')
    # 指定模型配置文件
    parser.add_argument('config', help='test config file path')
    # 模型權重資料
    parser.add_argument('checkpoint', help='checkpoint file')
    # 圖像保存地址
    parser.add_argument(
        '--save-path',
        type=str,
        default='./work_dirs/demos/unconditional_samples.png',
        help='path to save unconditional samples')
    # 推理設備
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device id')

    # args for inference/sampling
    # 總共有多少個batch
    parser.add_argument('--num-batches', type=int, default=4, help='Batch size in inference')
    # 總共需要產出多少張圖像
    parser.add_argument(
        '--num-samples',
        type=int,
        default=12,
        help='The total number of samples')
    # 初始噪聲產生方式
    parser.add_argument(
        '--sample-model',
        type=str,
        default='',
        help='Which model to use for sampling')
    # 額外添加到config的資料
    parser.add_argument(
        '--sample-cfg',
        nargs='+',
        action=DictAction,
        help='Other customized kwargs for sampling function')

    # args for image grid
    # 如果有需要進行padding就會用多少來進行
    parser.add_argument('--padding', type=int, default=0, help='Padding in the image grid.')
    # 每一行會有多少張圖像
    parser.add_argument(
        '--nrow',
        type=int,
        default=6,
        help='Number of images displayed in each row of the grid')

    args = parser.parse_args()
    return args


def main():
    # 進行圖像生成
    # 獲取啟動時傳入的參數
    args = parse_args()
    # 構建模型並且載入訓練權重
    model = init_model(args.config, checkpoint=args.checkpoint, device=args.device)

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    # 透過模型進行創造圖像，results = tensor shape [num_samples, channel, height, width]
    results = sample_unconditional_model(model, args.num_samples,
                                         args.num_batches, args.sample_model,
                                         **args.sample_cfg)
    # 將results當中的channel部分每個值都加一之後除以二，因為最後的激活函數會選擇tanh會將值控制在[-1, 1]之間
    # 所以透過轉換後值會在[0, 1]之間
    if results.shape[1] == 3:
        # 給輸出為RGB圖像使用
        results = (results[:, [2, 1, 0]] + 1.) / 2.
    elif results.shape[1] == 1:
        # 給輸出為灰階圖像使用
        results = (results[:, [0]] + 1.) / 2.

    # save images
    # 檢查存放圖像是否合法
    mmcv.mkdir_or_exist(os.path.dirname(args.save_path))
    # 進行圖像保存
    utils.save_image(results, args.save_path, nrow=args.nrow, padding=args.padding)


if __name__ == '__main__':
    main()
