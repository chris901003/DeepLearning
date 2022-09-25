from torch import nn
import torch
import numpy as np
from .weight_init import kaiming_init, constant_init
from .basic import BaseConv, DWConv, ConvModule
from .layer import Focus, CSPLayer, SPPBottleneck, BasicBlock3d, Bottleneck3d


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=('dark3', 'dark4', 'dark5'), depthwise=False, act='silu'):
        super(CSPDarknet, self).__init__()
        assert out_features
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        base_channel = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        self.stem = Focus(3, base_channel, ksize=3, act=act)
        self.dark2 = nn.Sequential(Conv(base_channel, base_channel * 2, 3, 2, act=act),
                                   CSPLayer(base_channel * 2, base_channel * 2, n=base_depth,
                                            depthwise=depthwise, act=act))
        self.dark3 = nn.Sequential(Conv(base_channel * 2, base_channel * 4, 3, 2, act=act),
                                   CSPLayer(base_channel * 4, base_channel * 4, n=base_depth * 3,
                                            depthwise=depthwise, act=act))
        self.dark4 = nn.Sequential(Conv(base_channel * 4, base_channel * 8, 3, 2, act=act),
                                   CSPLayer(base_channel * 8, base_channel * 8, n=base_depth * 3,
                                            depthwise=depthwise, act=act))
        self.dark5 = nn.Sequential(Conv(base_channel * 8, base_channel * 16, 3, 2, act=act),
                                   SPPBottleneck(base_channel * 16, base_channel * 16, activation=act),
                                   CSPLayer(base_channel * 16, base_channel * 16, n=base_depth, shortcut=False,
                                            depthwise=depthwise, act=act))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs['stem'] = x
        x = self.dark2(x)
        outputs['dark2'] = x
        x = self.dark3(x)
        outputs['dark3'] = x
        x = self.dark4(x)
        outputs['dark4'] = x
        x = self.dark5(x)
        outputs['dark5'] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class ResNet3d(nn.Module):
    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self, depth, pretrained, stage_blocks=None, pretrained2d=True, in_channels=3, num_stages=4,
                 base_channels=64, out_indices=(3, ), spatial_strides=(1, 2, 2, 2), temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1), conv1_kernel=(3, 7, 7), conv1_stride_s=2, conv1_stride_t=1, pool1_stride_s=2,
                 pool1_stride_t=1, with_pool1=True, with_pool2=True, frozen_stages=-1, inflate=(1, 1, 1, 1),
                 conv_cfg='Default', norm_cfg='Default', act_cfg='Default', norm_eval=False, zero_init_residual=True):
        super(ResNet3d, self).__init__()
        if conv_cfg == 'Default':
            conv_cfg = dict(type='Conv3d')
        if norm_cfg == 'Default':
            norm_cfg = dict(type='BN3d', requires_grad=True)
        if act_cfg == 'Default':
            act_cfg = dict(type='ReLU', inplace=True)
        if depth not in self.arch_settings:
            raise KeyError(f'Invalid depth {depth} for resnet')
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.stage_blocks = stage_blocks
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        if self.stage_blocks is not None:
            assert len(self.stage_blocks) == num_stages
        self.conv1_kernel = conv1_kernel
        self.conv1_stride_s = conv1_stride_s
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_s = pool1_stride_s
        self.pool1_stride_t = pool1_stride_t
        self.with_pool1 = with_pool1
        self.with_pool2 = with_pool2
        self.frozen_stages = frozen_stages
        self.stage_inflations = inflate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        if self.stage_blocks is None:
            self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = self.base_channels
        self._make_stem_layer()
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(self.block, self.inplanes, planes, num_blocks,
                                            spatial_stride=spatial_stride, temporal_stride=temporal_stride,
                                            dilation=dilation, norm_cfg=self.norm_cfg, conv_cfg=self.conv_cfg,
                                            act_cfg=self.act_cfg, inflate=self.stage_inflations[i])
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self.feat_dim = self.block.expansion * self.base_channels * 2 ** (len(self.stage_blocks) - 1)

    @staticmethod
    def make_res_layer(block, inplanes, planes, blocks, spatial_stride=1, temporal_stride=1, dilation=1, inflate=1,
                       norm_cfg=None, act_cfg=None, conv_cfg=None):
        inflate = inflate if not isinstance(inflate, int) else (inflate,) * blocks
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(inplanes, planes * block.expansion, kernel_size=1,
                                    stride=(temporal_stride, spatial_stride, spatial_stride), bias=False,
                                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        layers = list()
        layers.append(
            block(inplanes, planes, spatial_stride=spatial_stride, temporal_stride=temporal_stride, dilation=dilation,
                  downsample=downsample, inflate=(inflate[0] == 1),
                  norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(inplanes, planes, spatial_stride=1, temporal_stride=1, dilation=dilation,
                      inflate=(inflate[i] == 1), norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg))
        return nn.Sequential(*layers)

    def _make_stem_layer(self):
        self.conv1 = ConvModule(self.in_channels, self.base_channels, kernel_size=self.conv1_kernel,
                                stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
                                padding=tuple([(k - 1) // 2 for k in self.conv1_kernel]), bias=False,
                                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(self.pool1_stride_t, self.pool1_stride_s, self.conv1_stride_s),
                                    padding=(0, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        if self.with_pool1:
            x = self.maxpool(x)
        outs = list()
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def inflate_weights(self):
        self._inflate_weights(self)

    def init_weights(self, pretrained=None):
        self._init_weights(self, pretrained)

    @staticmethod
    def _inflate_conv_params(conv3d, state_dict_2d, module_name_2d, inflated_param_names):
        weight_2d_name = module_name_2d + '.weight'
        conv2d_weight = state_dict_2d[weight_2d_name]
        kernel_t = conv3d.weight.data.shape[2]
        new_weight = conv2d_weight.data.unsqueeze(dim=2).expand_as(conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)
        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    @staticmethod
    def _inflate_bn_params(bn3d, state_dict_2d, module_name_2d, inflated_param_names):
        for param_name, param in bn3d.named_parameters():
            param_2d_name = f'{module_name_2d}.{param_name}'
            param_2d = state_dict_2d[param_2d_name]
            if param.data.shape != param_2d.shape:
                print(f'{param_2d_name}權重加載失敗，2d shape {param_2d.shape}，3d shape {param.data.shape}')
                return
            param.data.copy_(param_2d)
            inflated_param_names.append(param_2d_name)
        for param_name, param in bn3d.named_buffers():
            param_2d_name = f'{module_name_2d}.{param_name}'
            if param_2d_name in state_dict_2d:
                param_2d = state_dict_2d[param_2d_name]
                param.data.copy_(param_2d)
                inflated_param_names.append(param_2d_name)

    @staticmethod
    def _inflate_weights(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        state_dict_r2d = torch.load(self.pretrained, map_location=device)
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']
        inflated_param_names = list()
        not_load_warring = list()
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                if 'downsample' in name:
                    # 這部分是需要根據torchvision官方給的預訓練權重層結構名稱定義的
                    original_conv_name = name + '.0'
                    original_bn_name = name + '.1'
                else:
                    original_conv_name = name
                    original_bn_name = name.replace('conv', 'bn')
                if original_conv_name + '.weight' not in state_dict_r2d:
                    not_load_warring.append(original_conv_name + '.weight')
                else:
                    shape_2d = state_dict_r2d[original_conv_name + '.weight'].shape
                    shape_3d = module.conv.weight.data.shape
                    if shape_2d != shape_3d[:2] + shape_3d[3:]:
                        not_load_warring.append(original_conv_name + '.weight')
                    else:
                        self._inflate_conv_params(module.conv, state_dict_r2d, original_conv_name, inflated_param_names)
                if original_bn_name + '.weight' not in state_dict_r2d:
                    not_load_warring.append(original_bn_name + '.weight')
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d, original_bn_name, inflated_param_names)
        remaining_names = set(state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            print('從2D預訓練權重加載，沒有匹配的層結構，如果只有fc沒有加載是正常還有其他的就不正常')
            print(remaining_names)
        if not_load_warring:
            print("\nFail To Load Key:", str(not_load_warring)[:500],
                  "……\nFail To Load Key num:", len(not_load_warring))

    @staticmethod
    def _init_weights(self, pretrained=None):
        if pretrained is not None:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            if self.pretrained2d:
                self.inflate_weights()
            else:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model_dict = self.state_dict()
                pretrained_dict = torch.load(self.pretrained, map_location=device)
                load_key, no_load_key, temp_dict = [], [], {}
                for k, v in pretrained_dict.items():
                    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                        temp_dict[k] = v
                        load_key.append(k)
                    else:
                        no_load_key.append(k)
                model_dict.update(temp_dict)
                self.load_state_dict(model_dict)
                print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
                print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck3d):
                        constant_init(m.conv3.bn, 0)
                    elif isinstance(m, BasicBlock3d):
                        constant_init(m.conv2.bn, 0)
        else:
            raise TypeError('初始化權重方式錯誤')
