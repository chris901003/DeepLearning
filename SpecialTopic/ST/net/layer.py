import torch
from torch import nn
from .basic import BaseConv, DWConv, ConvModule
from ..build import build_activation, build_norm


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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='Default', drop=0.):
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
