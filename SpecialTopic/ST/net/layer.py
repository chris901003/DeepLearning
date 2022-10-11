import torch
from torch import nn
from typing import Union
import math
import torch.nn.functional as F
from SpecialTopic.ST.utils import to_2tuple
from .basic import BaseConv, DWConv, ConvModule
from ..build import build_activation, build_norm, build_conv, build_dropout


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(hidden_channels * 2, out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act)
                       for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        super(Bottleneck, self).__init__()
        hidden_channels = int(in_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


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
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        return self.conv2(x)


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


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, spatial_stride=1, temporal_stride=1, dilation=1, downsample=None, inflate=True,
                 conv_cfg='Default', norm_cfg='Default', act_cfg='Default'):
        super(BasicBlock3d, self).__init__()
        if conv_cfg == 'Default':
            conv_cfg = dict(type='Conv3d')
        if norm_cfg == 'Default':
            norm_cfg = dict(type='BN3d')
        if act_cfg == 'Default':
            act_cfg = dict(type='ReLU')
        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.inflate = inflate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv1_stride_s = spatial_stride
        self.conv2_stride_s = 1
        self.conv1_stride_t = temporal_stride
        self.conv2_stride_t = 1
        if self.inflate:
            conv1_kernel_size = (3, 3, 3)
            conv1_padding = (1, dilation, dilation)
            conv2_kernel_size = (3, 3, 3)
            conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 3, 3)
            conv1_padding = (0, dilation, dilation)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)
        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            padding=conv1_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )
        self.conv2 = ConvModule(
            planes,
            planes * self.expansion,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s, self.conv2_stride_s),
            padding=conv2_padding,
            bias=False,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=None
        )
        self.downsample = downsample
        self.relu = build_activation(self.act_cfg)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck3d(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, spatial_stride=1, temporal_stride=1, dilation=1, downsample=None, inflate=True,
                 conv_cfg='Default', norm_cfg='Default', act_cfg='Default'):
        super(Bottleneck3d, self).__init__()
        if conv_cfg == 'Default':
            conv_cfg = dict(type='Conv3d')
        if norm_cfg == 'Default':
            norm_cfg = dict(type='BN3d')
        if act_cfg == 'Default':
            act_cfg = dict(type='ReLU')
        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.inflate = inflate
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg

        self.conv1_stride_s = 1
        self.conv2_stride_s = spatial_stride
        self.conv1_stride_t = 1
        self.conv2_stride_t = temporal_stride
        if self.inflate:
            conv1_kernel_size = (3, 1, 1)
            conv1_padding = (1, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = ConvModule(inplanes, planes, conv1_kernel_size,
                                stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
                                padding=conv1_padding, bias=False,
                                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv2 = ConvModule(planes, planes, conv2_kernel_size,
                                stride=(self.conv2_stride_t, self.conv2_stride_s, self.conv2_stride_s),
                                padding=conv2_padding, bias=False,
                                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv3 = ConvModule(planes, planes * self.expansion, kernel_size=1, bias=False,
                                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=None)
        self.downsample = downsample
        self.relu = build_activation(self.act_cfg)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class BasicBlockResnet(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlockResnet, self).__init__()
        self.layer1 = ConvModule(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False,
                                 conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        self.layer2 = ConvModule(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False,
                                 conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN'), act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out += identity
        out = self.relu(out)
        return out


class BottleneckResnet(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(BottleneckResnet, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.layer1 = ConvModule(in_channel, width, kernel_size=1, stride=1, bias=False,
                                 conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        self.layer2 = ConvModule(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False,
                                 conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        self.layer3 = ConvModule(width, out_channel * self.expansion, kernel_size=1, stride=1, bias=False,
                                 conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN'), act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out += identity
        out = self.relu(out)
        return out


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        batch_size, channel, height, width = x.shape
        assert height == self.img_size[0] and width == self.img_size[1], \
            f'輸入圖像大小與網路設定不符'
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchEmbedNormal(nn.Module):
    def __init__(self, in_channels=3, embed_dims=768, kernel_size=16, stride=None, padding=0, conv_type='Default',
                 norm_cfg=None, bias=True):
        super(PatchEmbedNormal, self).__init__()
        if conv_type == 'Default':
            conv_type = dict(type='Conv')
        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.projection = build_conv(conv_type, in_channels=in_channels, out_channels=embed_dims,
                                     kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if norm_cfg is not None:
            self.norm = build_norm(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class VitBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
                 drop_path_ratio=0., act_layer='Default', norm_layer='Default'):
        super(VitBlock, self).__init__()
        if act_layer == 'Default':
            act_layer = dict(type='GELU')
        if norm_layer == 'Default':
            norm_layer = dict(type='LN')
        self.norm1 = build_norm(norm_layer, dim)[1]
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = build_norm(norm_layer, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        batch_size, num_patch, channel = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_patch, 3, self.num_heads,
                                  channel // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_patch, channel)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer: Union[str, dict] = 'Default',
                 drop=0.):
        super(Mlp, self).__init__()
        if act_layer == 'Default':
            act_layer = dict(type='GELU')
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class DropPath(nn.Module):
    # 目前只有在VIT上被使用到
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    @staticmethod
    def drop_path(x, drop_prob: float = 0., training: bool = False):
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


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, skip_connection=True,
                 conv_cfg='Default', norm_cfg='Default', act_cfg='Default'):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        if conv_cfg == 'Default':
            conv_cfg = dict(type='Conv')
        if norm_cfg == 'Default':
            norm_cfg = dict(type='BN')
        if act_cfg == 'Default':
            act_cfg = dict(type='SiLU')
        hidden_dim = self.make_divisible(int(round(in_channels * expand_ratio)), 8)
        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name='exp_1x1',
                module=ConvModule(
                    in_channels=in_channels, out_channels=hidden_dim, kernel_size=1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        block.add_module(
            name='conv_3x3',
            module=ConvModule(
                in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=3, padding=1,
                groups=hidden_dim, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        block.add_module(
            name='red_1x1',
            module=ConvModule(
                in_channels=hidden_dim, out_channels=out_channels, kernel_size=1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None))
        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels and skip_connection)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    @staticmethod
    def make_divisible(v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, num_heads=8, attn_dropout=0.0, dropout=0.0, ffn_dropout=0.0):
        super(TransformerEncoder, self).__init__()
        attn_uint = Attention(embed_dim, num_heads, qkv_bias=True,
                              attn_drop_ratio=attn_dropout, proj_drop_ratio=dropout)
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim),
            attn_uint
        )
        act_layer = dict(type='SiLU')
        mlp = Mlp(in_features=embed_dim, hidden_features=ffn_latent_dim, out_features=embed_dim,
                  act_layer=act_layer, drop=ffn_dropout)
        self.pre_norm_ffn = nn.Sequential(nn.LayerNorm(embed_dim), mlp)
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout

    def forward(self, x):
        res = x
        x = self.pre_norm_mha(x)
        x = x + res
        x = x + self.pre_norm_ffn(x)
        return x


class MobileVitBlock(nn.Module):
    def __init__(self, in_channels, transformer_dim, ffn_dim, transformer_blocks, head_dim=32, attn_dropout=0.0,
                 dropout=0.0, ffn_dropout=0.0, patch_h=8, patch_w=8, conv_ksize=3,
                 conv_cfg='Default', norm_cfg='Default', act_cfg='Default'):
        super(MobileVitBlock, self).__init__()
        if conv_cfg == 'Default':
            conv_cfg = dict(type='Conv')
        if norm_cfg == 'Default':
            norm_cfg = dict(type='BN')
        if act_cfg == 'Default':
            act_cfg = dict(type='SiLU')
        conv_3x3_in = ConvModule(in_channels=in_channels, out_channels=in_channels, kernel_size=conv_ksize, stride=1,
                                 padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        conv_1x1_in = ConvModule(in_channels=in_channels, out_channels=transformer_dim, kernel_size=1, bias=False,
                                 conv_cfg=conv_cfg, norm_cfg=None, act_cfg=None)
        conv_1x1_out = ConvModule(in_channels=transformer_dim, out_channels=in_channels, kernel_size=1, stride=1,
                                  conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        conv_3x3_out = ConvModule(in_channels=2 * in_channels, out_channels=in_channels, kernel_size=conv_ksize,
                                  padding=1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name='conv_3x3', module=conv_3x3_in)
        self.local_rep.add_module(name='conv_1x1', module=conv_1x1_in)
        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim
        global_rep = [
            TransformerEncoder(embed_dim=transformer_dim, ffn_latent_dim=ffn_dim, num_heads=num_heads,
                               attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout)
            for _ in range(transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)
        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x):
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape
        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            interpolate = True
        num_patch_w = new_w // patch_w
        num_patch_h = new_h // patch_h
        num_patches = num_patch_w * num_patch_h
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        x = x.transpose(1, 3)
        x = x.reshape(batch_size * patch_area, num_patches, -1)
        info_dict = {
            'orig_size': (orig_h, orig_w),
            'batch_size': batch_size,
            'interpolate': interpolate,
            'total_patches': num_patches,
            'num_patches_w': num_patch_w,
            'num_patches_h': num_patch_h
        }
        return x, info_dict

    def folding(self, x, info_dict):
        n_dim = x.dim()
        assert n_dim == 3, 'Tensor格式錯誤'
        x = x.contiguous().view(info_dict['batch_size'], self.patch_area, info_dict['total_patches'], -1)
        batch_size, pixels, num_patches, channels = x.shape
        num_patch_h = info_dict['num_patches_h']
        num_patch_w = info_dict['num_patches_w']
        x = x.transpose(1, 3)
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict['interpolate']:
            x = F.interpolate(x, size=info_dict['orig_size'], mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        res = x
        fm = self.local_rep(x)
        patches, info_dict = self.unfolding(fm)
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)
        fm = self.folding(x=patches, info_dict=info_dict)
        fm = self.conv_proj(fm)
        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., dropout_layer='Default', batch_first=False,
                 **kwargs):
        # 這裡比較特別的是直接使用pytorch官方給的注意力模塊，所以如果遇到預訓練權重是使用官方的attention模塊就用這個
        super(MultiheadAttention, self).__init__()
        if dropout_layer == 'Default':
            dropout_layer = dict(type='Dropout', p=0.)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_layer = build_dropout(dropout_layer) if dropout_layer is not None else nn.Identity()

    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None, attn_mask=None,
                key_padding_mask=None, **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
        if query_pos is not None:
            query += query_pos
        if key_pos is not None:
            key = key + key_pos
        if self.batch_first:
            query = query.tranpose(0, 1)
            key = key.transpose(0, 1)
            value = value.tranpose(0, 1)
        out = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.drop_layer(self.proj_drop(out))
