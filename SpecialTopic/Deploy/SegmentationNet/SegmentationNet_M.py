import argparse
import os
import torch
from torch import nn


class PatchEmbedNormal(nn.Module):
    def __init__(self, in_channels=3, embed_dims=768, kernel_size=16, stride=None, padding=0,
                 norm_cfg=None, bias=True):
        super(PatchEmbedNormal, self).__init__()
        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size
        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=embed_dims, kernel_size=kernel_size,
                                    stride=stride, padding=padding, bias=bias)
        if norm_cfg == 'LN':
            self.norm = nn.LayerNorm(embed_dims, eps=1e-6)
        else:
            self.norm = None
            raise NotImplementedError('尚未提供除了LN以外的歸一化層')

    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[2])
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, out_size
        

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
        in_channels = 3
        num_layers = (3, 4, 6, 3)
        num_heads = (1, 2, 4, 8)
        patch_sizes = (7, 3, 3, 3)
        strides = (4, 2, 2, 2)
        sr_ratios = (8, 4, 2, 1)
        mlp_ratio = 4
        # TODO: 如果轉成TensorRT過程中發生錯誤請先將這裡進行調整
        dpr = [x.item() for x in torch.linspace(0, 0.1, 16)]
        self.layers = nn.ModuleList()
        cur = 0
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = 64 * num_heads[i]
            patch_embed = PatchEmbedNormal(in_channels=in_channels, embed_dims=embed_dims_i, kernel_size=patch_sizes[i],
                                           stride=strides[i], padding=patch_sizes[i] // 2, norm_cfg='LN')
            layer = nn.ModuleList([
                TransformerEncoderLayer(embed_dims=embed_dims_i, num_heads=num_heads[i],
                                        feedforward_channels=mlp_ratio * embed_dims_i, drop_rate=0.,
                                        attn_drop_rate=0., drop_path_rate=dpr[cur + idx], qkv_bias=False,
                                        act_cfg='GELU', norm_cfg='LN', sr_ratio=sr_ratios[i])
                for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            norm = nn.LayerNorm(normalized_shape=embed_dims_i, eps=1e-6)
            block = nn.ModuleList([patch_embed, layer, norm])
            self.layers.append(block)
            cur += num_layer

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
