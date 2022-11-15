import tensorrt
import argparse
import torch
import torch.nn.functional as F
from torch import nn


def nlc_to_nchw(x, hw_shape):
    H, W = hw_shape
    batch_size, top_patch, channels = x.shape
    return x.transpose(1, 2).reshape(batch_size, channels, H, W)


def nchw_to_nlc(x):
    batch_size, channels, height, width = x.shape
    return x.reshape(batch_size, channels, -1).transpose(1, 2)


class DropPath(nn.Module):
    def __init__(self, drop_prob):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    @staticmethod
    def drop_path(x, drop_prob: float = 0., training: bool = False):
        # 由於是用來轉成onnx，所以正常來說會直接回傳x，也就是DropPath結構不起作用
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_cfg=True,
                 act_cfg=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        if norm_cfg:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        if act_cfg:
            self.activate = nn.ReLU(inplace=True)
        else:
            self.activate = None

    def forward(self, x):
        out = self.activate(self.bn(self.conv(x)))
        return out


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


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., dropout_layer=None, bias=True):
        super(MultiheadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        if isinstance(dropout_layer, dict):
            if dropout_layer['type'] == 'DropPath':
                # 在驗證模式下DropPath不會啟動，所以這裡直接使用Identity取代
                self.drop_layer = nn.Identity()
            else:
                NotImplementedError('目前只支援DropPath')
        else:
            raise NotImplementedError('dropout_layer目前需要是dict')

    def forward(self, x, hw_shape, identity=None):
        raise NotImplementedError('這裡只提供繼承，不進行實作')


class EfficientMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., dropout_layer=None, qkv_bias=False,
                 norm_cfg='LN', sr_ratio=1):
        super(EfficientMultiheadAttention, self).__init__(embed_dims, num_heads, attn_drop, proj_drop,
                                                          dropout_layer=dropout_layer, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=sr_ratio, stride=sr_ratio)
            if norm_cfg == 'LN':
                self.norm = nn.LayerNorm(embed_dims, eps=1e-6)
            else:
                self.norm = None
                raise NotImplementedError('目前只提供LN的標準化層結構')

    def forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
        if identity is None:
            identity = x_q
        x_q = x_q.transpose(0, 1)
        x_kv = x_kv.transpose(0, 1)
        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]
        out = out.transpose(0, 1)
        return identity + self.drop_layer(self.proj_drop(out))


class MixFFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, act_cfg='GELU', ffn_drop=0.):
        super(MixFFN, self).__init__()
        if act_cfg == 'GELU':
            activate = nn.GELU()
        else:
            raise NotImplementedError('目前只支援GELU作為激活函數')
        fc1 = nn.Conv2d(in_channels=embed_dims, out_channels=feedforward_channels, kernel_size=1, stride=1, bias=True)
        pe_conv = nn.Conv2d(in_channels=feedforward_channels, out_channels=feedforward_channels, kernel_size=3,
                            stride=1, padding=1, bias=True, groups=feedforward_channels)
        fc2 = nn.Conv2d(in_channels=feedforward_channels, out_channels=embed_dims, kernel_size=1, stride=1, bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, feedforward_channels, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 qkv_bias=True, act_cfg='GELU', norm_cfg='LN', sr_ratio=1):
        super(TransformerEncoderLayer, self).__init__()
        if norm_cfg == 'LN':
            self.norm1 = nn.LayerNorm(embed_dims, eps=1e-6)
        else:
            raise NotImplementedError('目前提供的標準化層只有LN')
        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims, num_heads=num_heads, attn_drop=attn_drop_rate, proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate), qkv_bias=qkv_bias, norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.ffn = MixFFN(
            embed_dims=embed_dims, feedforward_channels=feedforward_channels, ffn_drop=drop_rate,
            act_cfg=act_cfg)

    def forward(self, x, hw_shape):
        x = self.attn(self.norm1(x), hw_shape, identity=x)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        return x


class BaseDecodeHead(nn.Module):
    def __init__(self, in_channels, channels, *, num_classes, dropout_ratio=0.1, in_index=(0, 1, 2, 3),
                 input_transform=None, align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self.channels = channels
        self.input_transform = input_transform
        self.in_index = in_index
        self.in_channels = in_channels
        self.align_corners = align_corners
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def _transform_inputs(self, inputs):
        inputs = [inputs[i] for i in self.in_index]
        return inputs

    def forward(self, **kwargs):
        pass


class SegformerHead(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super(SegformerHead, self).__init__(input_transform=interpolate_mode, **kwargs)
        num_inputs = len(self.in_channels)
        self.convs = nn.ModuleList()
        self.interpolate_mode = interpolate_mode
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(in_channels=self.in_channels[i], out_channels=self.channels, kernel_size=1, stride=1))
        self.fusion_conv = ConvModule(in_channels=self.channels * num_inputs, out_channels=self.channels, kernel_size=1)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = list()
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                F.interpolate(input=conv(x), size=inputs[0].shape[2:], mode=self.interpolate_mode,
                              align_corners=self.align_corners)
            )
        out = self.fusion_conv(torch.cat(outs, dim=1))
        seg_pred = self.cls_seg(out)
        # seg_pred = F.interpolate(input=out, size=self.image_size, mode='bilinear', align_corners=False)
        return seg_pred
        

class SegmentationNetM(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationNetM, self).__init__()
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
        num_heads = (1, 2, 5, 8)
        patch_sizes = (7, 3, 3, 3)
        strides = (4, 2, 2, 2)
        sr_ratios = (8, 4, 2, 1)
        self.out_indices = (0, 1, 2, 3)
        mlp_ratio = 4
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
        decode_head_cfg = {
            'in_channels': [64, 128, 320, 512],
            'in_index': [0, 1, 2, 3],
            'channels': 256,
            'dropout_ratio': 0.1,
            'num_classes': num_classes,
        }
        self.decode_head = SegformerHead(**decode_head_cfg)

    def forward(self, x):
        # image_height, image_width = x.shape[2:]
        # self.decode_head.image_size = image_height, image_width
        outs = list()
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)
        outs = self.decode_head(outs)
        return outs


def load_pretrained(model, pretrained_path):
    import os
    import numpy as np
    assert os.path.exists(pretrained_path), '提供的模型權重不存在'
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    if 'model_weight' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_weight']
    load_key, no_load_key, temp_dict = list(), list(), dict()
    for k, v in pretrained_dict.items():
        idx = k.find('.')
        if k[:idx] == 'backbone':
            new_name = k[idx + 1:]
        else:
            new_name = k
        if new_name in model_dict.keys() and np.shape(model_dict[new_name]) == np.shape(v):
            temp_dict[new_name] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    assert len(no_load_key) == 0, '給定的預訓練權重與模型不匹配'
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    # 分割類別數
    parser.add_argument('--num-classes', type=int, default=3)
    # 訓練權重路徑
    parser.add_argument('--pretrained', type=str, default=r'C:\Checkpoint\SegformerFoodAndNot'
                                                          r'FoodDetection\1024_eval_0.pth')
    args = parser.parse_args()
    return args


def main():
    """ 主要是生成Segformer模型大小為m的onnx檔案資料
    """
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SegmentationNetM(num_classes=args.num_classes)
    model = load_pretrained(model, args.pretrained)
    model.eval()
    model = model.to(device)
    images = torch.randn(1, 3, 512, 512).to(device)
    preds = model(images)
    print(preds.shape)
    dynamic_axes = {'images_seg': {2: 'image_height', 3: 'image_width'}}
    input_names = ['images_seg']
    output_names = ['outputs_seg']
    with torch.no_grad():
        torch.onnx.export(model, images, 'SegmentationNetM.onnx', input_names=input_names,
                          output_names=output_names, opset_version=11, dynamic_axes=dynamic_axes)


def test_onnx_file():
    import onnxruntime
    onnx_file = 'SegmentationNetM.onnx'
    session = onnxruntime.InferenceSession(onnx_file, providers=['CUDAExecutionProvider'])
    x = torch.randn(1, 3, 512, 512)
    onnx_inputs = {'images_seg': x.cpu().numpy()}
    onnx_outputs = ['outputs_seg']
    outputs = session.run(onnx_outputs, onnx_inputs)[0]
    print(outputs.shape)


def simplify_onnx():
    from onnxsim import simplify
    import onnx
    onnx_path = 'SegmentationNetM.onnx'
    output_path = 'SegmentationNetM_Simplify.onnx'
    onnx_model = onnx.load(onnx_path)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    print('finished exporting onnx')


def test_tensorrt():
    from SpecialTopic.Deploy.OnnxToTensorRT.TensorrtBase import TensorrtBase
    from SpecialTopic.ST.dataset.utils import Compose
    import cv2
    import numpy as np
    from PIL import Image
    onnx_file_path = 'SegmentationNetM_Simplify.onnx'
    trt_engine_path = 'SegmentationNetM.trt'
    save_trt_engine_path = 'SegmentationNetM.trt'
    # save_trt_engine_path = None
    trt_logger_level = 'VERBOSE'
    fp16_mode = True
    max_batch_size = 1
    # 這裡即使我們將輸出的shape設定成固定shape但是onnx資料還是會自動將輸出資料變成動態shape
    # 由於我們可以肯定輸出的大小，所以在這裡先將輸出的記憶體空間大小固定下來，才比較容易進行控制
    # 否則一方面是記憶體不好管控，另一方面是最後reshape輸出時會有困難
    dynamic_shapes = {'images_seg': ((1, 3, 512, 512), (1, 3, 512, 512), (1, 3, 800, 800)),
                      'outputs_seg': ((1, 3, 128, 128), (1, 3, 128, 128), (1, 3, 128, 128))}
    tensorrt_engine = TensorrtBase(onnx_file_path=onnx_file_path, fp16_mode=fp16_mode, max_batch_size=max_batch_size,
                                   dynamic_shapes=dynamic_shapes, save_trt_engine_path=save_trt_engine_path,
                                   trt_engine_path=trt_engine_path, trt_logger_level=trt_logger_level)
    image_path = r'C:\Dataset\SegmentationFoodRemain\Donburi\images\training\1.jpg'
    pipeline_cfg = [
        {'type': 'MultiScaleFlipAugSegformer', 'img_scale': (2048, 512), 'flip': False,
         'transforms': [
             {'type': 'ResizeMMlab', 'keep_ratio': True},
             {'type': 'RandomFlipMMlab'},
             {'type': 'NormalizeMMlab', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375],
              'to_rgb': True},
             {'type': 'ImageToTensorMMlab', 'keys': ['img']},
             {'type': 'Collect', 'keys': ['img']}
         ]}
    ]
    pipeline = Compose(pipeline_cfg)
    image = cv2.imread(image_path)
    origin_image = image.copy()
    image = cv2.resize(image, (512, 512))
    data = dict(img=image)
    data = pipeline(data)
    image = data['img'][0].unsqueeze(dim=0).numpy()
    trt_input = {'images_seg': np.ascontiguousarray(image).astype(np.float32)}
    trt_output = ['outputs_seg']
    output = tensorrt_engine.inference(input_datas=trt_input, output_shapes=trt_output,
                                       dynamic_shape=True)[0]
    output = torch.from_numpy(output)
    # 最後還需要將插值部分移出模型當中，後處理再進行，因為這樣會使的輸出大小不確定導致不好處理記憶體空間申請，以及調整
    output = F.interpolate(input=output, size=image.shape[2:], mode='bilinear', align_corners=False)
    seg_pred = F.interpolate(input=output, size=origin_image.shape[:2], mode='bilinear', align_corners=False)
    seg_pred = F.softmax(seg_pred, dim=1)
    mask = (seg_pred > 0.8).squeeze(dim=0).cpu().numpy()
    seg_pred = seg_pred.argmax(dim=1)
    seg_pred = seg_pred.cpu().numpy()[0]
    from SpecialTopic.ST.dataset.config.segmentation_classes_platte import FoodAndNotFood
    CLASSES = FoodAndNotFood['CLASSES']
    PALETTE = FoodAndNotFood['PALETTE']
    draw_image_mix, draw_image = image_draw(origin_image, seg_pred, palette=PALETTE, classes=CLASSES,
                                            opacity=0.5, with_class=False, mask=mask)
    img = Image.fromarray(cv2.cvtColor(draw_image_mix, cv2.COLOR_BGR2RGB))
    img.show()

    from SpecialTopic.SegmentationNet.api import init_module, detect_single_picture
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch_model = init_module(model_type='Segformer', phi='m',
                              pretrained=r'C:\Checkpoint\SegformerFoodAndNotFoodDetection\1024_eval_0.pth',
                              device=device, with_color_platte='FoodAndNotFood', num_classes=3)
    results = detect_single_picture(torch_model, device, origin_image, with_class=True, threshold=0.8)
    draw_image_mix = results[0]
    imgs = Image.fromarray(cv2.cvtColor(draw_image_mix, cv2.COLOR_BGR2RGB))
    imgs.show()


def image_draw(origin_image, seg_pred, palette, classes, opacity, with_class=False, mask=None):
    import copy
    import numpy as np
    import cv2

    image = copy.deepcopy(origin_image)
    assert palette is not None, '需要提供調色盤，否則無法作畫'
    palette = np.array(palette)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[(seg_pred == label) & mask[label], :] = color
    color_seg = color_seg[..., ::-1]
    image = image * (1 - opacity) + color_seg * opacity
    image = image.astype(np.uint8)
    if with_class:
        classes_set = set(seg_pred.flatten())
        color2classes = {classes[idx]: palette[idx] for idx in classes_set}
        image_height, image_width = image.shape[:2]
        ymin, ymax = image_height - 30, image_height - 10
        for idx, (class_name, color) in enumerate(color2classes.items()):
            cv2.rectangle(image, (20, ymin), (40, ymax), color[::-1].tolist(), -1)
            cv2.putText(image, class_name, (50, ymax), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (89, 214, 210), 2, cv2.LINE_AA)
            ymin -= 30
            ymax -= 30
    return image, color_seg


if __name__ == '__main__':
    print('Starting create Segmentation net [m] onnx')
    # main()
    # simplify_onnx()
    test_tensorrt()
