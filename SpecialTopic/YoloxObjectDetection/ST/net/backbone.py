from torch import nn
import torch
from .basic import BaseConv, DWConv
from .layer import Focus, CSPLayer, SPPBottleneck


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
