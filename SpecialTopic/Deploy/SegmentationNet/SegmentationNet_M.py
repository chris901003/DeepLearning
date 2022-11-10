import argparse
import os
import torch
from torch import nn


class Segformer(nn.Module):
    def __init__(self, num_classes):
        super(Segformer, self).__init__()
        # 此部分的參數都是模型大小為m的資料
        self.num_classes = num_classes

        # 構建backbone部分
        """
        in_channel = 3
        embed_dims = 64
        num_stages = 4
        num_layers = [3, 4, 6, 3]
        num_heads = [1, 2, 5, 8]
        patch_sizes = [7, 3, 3, 3]
        sr_ratios = [8, 4, 2, 1]
        out_indices = (0, 1, 2, 3)
        mlp_ratio = 4
        qkv_bias = True
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.1
        act_type = GELU
        norm_type = LN, eps=1e-6
        """
        self.layers = nn.ModuleList()
        # TODO: 如果轉成TensorRT過程中發生錯誤請先將這裡進行調整
        dpr = [x.item() for x in torch.linspace(0, 0.1, 16)]

    def forward(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    # 分割類別數
    parser.add_argument('--num-classes', type=int, default=3)
    # 訓練權重路徑
    parser.add_argument('--pretrained', type=str, default='pretrained.pth')
    args = parser.parse_args()
    return args


def main():
    """ 主要是生成Segformer模型大小為m的onnx檔案資料
    """
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Segformer(num_classes=args.num_classes)
    model = load_pretrained(model, args.pretrained)
    model.eval()
    model = model.to(device)
    images = torch.randn(1, 3, 512, 512).to(device)
    input_names = ['images_seg']
    output_names = ['outputs_seg']
    with torch.no_grad():
        model_script = torch.jit.script(model)
        torch.onnx.export(model_script, images, 'SegmentationNetM.onnx', input_names=input_names,
                          output_names=output_names, opset_version=11)


if __name__ == '__main__':
    print('Starting create Segmentation net [m] onnx')
    main()
