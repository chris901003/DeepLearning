import torch
from torch import nn
import os
import numpy as np
import argparse


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, bias=False, act='silu'):
        super(BaseConv, self).__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        if act == 'silu':
            self.act = nn.SiLU()
        else:
            print('Base Conv: 未獲取有效激活函數類型')
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, act='silu'):
        super(Bottleneck, self).__init__()
        hidden_channels = int(in_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, act='silu'):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(hidden_channels * 2, out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, act=act)
                       for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act='silu'):
        super(Focus, self).__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 9, 13), activation='silu'):
        super(SPPBottleneck, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_size])
        conv2_channels = hidden_channels * (len(kernel_size) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.m[0](x)
        x2 = self.m[1](x)
        x3 = self.m[2](x)
        x = torch.cat([x, x1, x2, x3], dim=1)
        # spp_list = list([x])
        # for m in self.m:
        #     spp_list.append(m(x))
        # x = torch.cat(spp_list, dim=1)
        # x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        return self.conv2(x)


class CSPDarknet(nn.Module):
    def __init__(self):
        super(CSPDarknet, self).__init__()
        base_channel, base_depth = 64, 3
        self.stem = Focus(3, base_channel, ksize=3, act='silu')
        self.dark2 = nn.Sequential(BaseConv(base_channel, base_channel * 2, 3, 2, act='silu'),
                                   CSPLayer(base_channel * 2, base_channel * 2, n=base_depth, act='silu'))
        self.dark3 = nn.Sequential(BaseConv(base_channel * 2, base_channel * 4, 3, 2, act='silu'),
                                   CSPLayer(base_channel * 4, base_channel * 4, n=base_depth * 3, act='silu'))
        self.dark4 = nn.Sequential(BaseConv(base_channel * 4, base_channel * 8, 3, 2, act='silu'),
                                   CSPLayer(base_channel * 8, base_channel * 8, n=base_depth * 3, act='silu'))
        self.dark5 = nn.Sequential(BaseConv(base_channel * 8, base_channel * 16, 3, 2, act='silu'),
                                   SPPBottleneck(base_channel * 16, base_channel * 16, activation='silu'),
                                   CSPLayer(base_channel * 16, base_channel * 16, n=base_depth, shortcut=False,
                                            act='silu'))

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        d3 = self.dark3(x)
        d4 = self.dark4(d3)
        d5 = self.dark5(d4)
        outputs = [d3, d4, d5]
        return outputs


class YoloxObjectDetection(nn.Module):
    def __init__(self, num_classes):
        """ 本檔案提供的是模型大小為L的類別
        Args:
            num_classes: 分類類別數
        """
        super(YoloxObjectDetection, self).__init__()
        """ 提供對應到正常訓練下的配置，本檔案提供的是模型大小為L的類別
        backbone_cfg = {
            'type': 'YOLOPAFPN',
            'depth': 1,
            'width': 1,
            'depthwise': False
        } 
        head_cfg = {
            'type': 'YOLOXHead',
            'num_classes': num_classes,
            'depth': 1,
            'width': 1,
            'depthwise': False
        }
        """
        # YoloPAFPN
        self.backbone = CSPDarknet()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv0 = BaseConv(1024, 512, 1, 1, act='silu')
        self.C3_p4 = CSPLayer(1024, 512, 3, False, act='silu')
        self.reduce_conv1 = BaseConv(512, 256, 1, 1, act='silu')
        self.C3_p3 = CSPLayer(512, 256, 3, False, act='silu')
        self.bu_conv2 = BaseConv(256, 256, 3, 2, act='silu')
        self.C3_n3 = CSPLayer(512, 512, 3, False, act='silu')
        self.bu_conv1 = BaseConv(512, 512, 3, 2, act='silu')
        self.C3_n4 = CSPLayer(1024, 1024, 3, False, act='silu')

        # YoloXHead
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        in_channels = [256, 512, 1024]
        for in_channel in in_channels:
            self.stems.append(BaseConv(in_channels=in_channel, out_channels=256, ksize=1, stride=1, act='silu'))
            self.cls_convs.append(nn.Sequential(*[
                BaseConv(in_channels=256, out_channels=256, ksize=3, stride=1, act='silu'),
                BaseConv(in_channels=256, out_channels=256, ksize=3, stride=1, act='silu')
            ]))
            self.cls_preds.append(nn.Conv2d(256, out_channels=num_classes, kernel_size=1, stride=1, padding=0))
            self.reg_convs.append(nn.Sequential(*[
                BaseConv(in_channels=256, out_channels=256, ksize=3, stride=1, act='silu'),
                BaseConv(in_channels=256, out_channels=256, ksize=3, stride=1, act='silu')
            ]))
            self.reg_preds.append(nn.Conv2d(in_channels=256, out_channels=4, kernel_size=1, stride=1, padding=0))
            self.obj_preds.append(nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0))

    def forward(self, inputs):
        out_features = self.backbone(inputs)
        feat1, feat2, feat3 = out_features
        P5 = self.lateral_conv0(feat3)
        P5_upsample = self.upsample(P5)
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P5_upsample = self.C3_p4(P5_upsample)
        P4 = self.reduce_conv1(P5_upsample)
        P4_upsample = self.upsample(P4)
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        P3_out = self.C3_p3(P4_upsample)
        P3_downsample = self.bu_conv2(P3_out)
        P3_downsample = torch.cat([P3_downsample, P4], 1)
        P4_out = self.C3_n3(P3_downsample)
        P4_downsample = self.bu_conv1(P4_out)
        P4_downsample = torch.cat([P4_downsample, P5], 1)
        P5_out = self.C3_n4(P4_downsample)

        # outputs = list()
        idx = 0
        fpn_inputs = [P3_out, P4_out, P5_out]
        output1, output2, output3 = fpn_inputs
        for stem, cls_conv, cls_pred, reg_conv, reg_pred, obj_pred in zip(
                self.stems, self.cls_convs, self.cls_preds, self.reg_convs, self.reg_preds, self.obj_preds):
            x = stem(fpn_inputs[idx])
            cls_feat = cls_conv(x)
            cls_output = cls_pred(cls_feat)
            reg_feat = reg_conv(x)
            reg_output = reg_pred(reg_feat)
            obj_output = obj_pred(reg_feat)
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            # outputs.append(output)
            if idx == 0:
                output1 = output
            elif idx == 1:
                output2 = output
            else:
                output3 = output
            idx += 1
        # return outputs
        return output1, output2, output3


def load_pretrained(model, pretrained_path):
    assert os.path.exists(pretrained_path), '需提供預訓練權重資料'
    print(f'Load weights {pretrained_path}')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    if 'model_weight' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_weight']
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        idx = k.find('.')
        new_name = k[idx + 1:]
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
    # 分類類別數
    parser.add_argument('--num-classes', type=int, default=9)
    # 訓練權重資料位置，這裡一定要加載進去
    parser.add_argument('--pretrained', type=str, default=r'C:\Checkpoint\YoloxFoodDetection\900_yolox_850.25.pth')
    args = parser.parse_args()
    return args


def main():
    """
    主要是生成Yolox Object Detection模型大小為L的onnx模型檔案
    如果生成設備上有gpu就會產生出可以支持gpu版本的onnx檔案
    """
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YoloxObjectDetection(num_classes=args.num_classes)
    model = load_pretrained(model, args.pretrained)
    model.eval()
    model = model.to(device)
    images = torch.randn(1, 3, 640, 640).to(device)
    # preds = model(images)
    input_names = ['images']
    output_names = ['outputs']
    with torch.no_grad():
        model_script = torch.jit.script(model)
        torch.onnx.export(model_script, images, 'YoloxObjectDetectionL.onnx', input_names=input_names,
                           output_names=output_names, opset_version=11)


if __name__ == '__main__':
    print('Starting create Yolox object detection [l] onnx')
    main()
