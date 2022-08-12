# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_plugin_layer


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def conv1x1(in_planes, out_planes):
    """1x1 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 use_conv1x1=False,
                 plugins=None):
        """ 已看過，基礎的ResNet層結構初始化部分
        Args:
            inplanes: 輸入的channel深度
            planes: 輸出的channel深度
            stride: 步距
            downsample: 殘差結構的下採樣方式
            use_conv1x1: 是否使用1*1的卷積，原始的ResNet在兩個卷積都是使用3*3卷積
            plugins: 額外的插入層結構
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super(BasicBlock, self).__init__()

        if use_conv1x1:
            # 如果要使用1*1卷積就會到這裡，將第一個卷積改成1*1卷積
            self.conv1 = conv1x1(inplanes, planes)
            self.conv2 = conv3x3(planes, planes * self.expansion, stride)
        else:
            # 通過第一個卷積進行擴維
            self.conv1 = conv3x3(inplanes, planes, stride)
            # 這裡的expansion會是1，所以通過第二個卷積後channel不會改變
            self.conv2 = conv3x3(planes, planes * self.expansion)

        # 先將with_plugins設定成False
        self.with_plugins = False
        if plugins:
            # 如果有傳入plugins就會到這裡
            if isinstance(plugins, dict):
                # 如果傳入的是dict就用list將其包起來
                plugins = [plugins]
            # 將with_plugins設定成True
            self.with_plugins = True

            # collect plugins for conv1/conv2/
            # 如果是在conv1前經過的會到這裡
            self.before_conv1_plugin = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'before_conv1'
            ]
            # 如果是在conv1後conv2前的會到這裡
            self.after_conv1_plugin = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            # 如果是在conv2後的會到這裡
            self.after_conv2_plugin = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            # 如果是接在殘差邊相加後的會到這裡
            self.after_shortcut_plugin = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_shortcut'
            ]

        # 將輸出的channel深度進行保留
        self.planes = planes
        # 構建BN標準化層結構，給第一個卷積出來的
        self.bn1 = nn.BatchNorm2d(planes)
        # 構建激活函數
        self.relu = nn.ReLU(inplace=True)
        # 構建BN標準化層結構，給第二個卷積出來的
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        # 保存下採樣的層結構
        self.downsample = downsample
        # 保存步距
        self.stride = stride

        if self.with_plugins:
            # 如果有需要plugins的層結構就會到這裡進行實例化
            # 將輸入的channel以及config資料傳入
            self.before_conv1_plugin_names = self.make_block_plugins(
                inplanes, self.before_conv1_plugin)
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugin)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugin)
            self.after_shortcut_plugin_names = self.make_block_plugins(
                planes, self.after_shortcut_plugin)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        # 已看過，構建插入的層結構
        # in_channels = 輸入的channel深度
        # plugins = 需要插入的層結構設定資料

        # 檢查plugins是否為list型態
        assert isinstance(plugins, list)
        # 構建保存實例化對象的對應名稱，因為我們使用add_module到模型當中，需要時是用名稱進行呼叫
        plugin_names = []
        # 遍歷所有plugins的config資料
        for plugin in plugins:
            # 複製一份config資料
            plugin = plugin.copy()
            # 透過build_plugin_layer將需要插入的層結構進行實例化
            name, layer = build_plugin_layer(
                # 將層結構的config文件傳入
                plugin,
                # 傳入輸入channel深度與輸出channel深度
                in_channels=in_channels,
                out_channels=in_channels,
                # 命名時會需要用到的
                postfix=plugin.pop('postfix', ''))
            # 檢查在當前模型下是否有相同名稱的實例對象
            assert not hasattr(self, name), f'duplicate plugin {name}'
            # 使用add_module將層結構放到模型當中
            self.add_module(name, layer)
            # 將名稱保留下來
            plugin_names.append(name)
        # 回傳名稱列表
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        """ 已看過，中間插入的層結構會到這裡
        Args:
            x: 特徵圖，這裡的shape會根據要通過的層結構會有所不同
            plugin_names: 要通過的層結構名稱
        """
        # 保存傳入的特徵層名稱
        out = x
        # 遍歷要通過的層結構
        for name in plugin_names:
            # 找到模型當中對應名稱的層結構實例化對象，並且進行正向傳播
            out = getattr(self, name)(x)
        # 最後將結果輸出
        return out

    def forward(self, x):
        """ 已看過，ResNet層結構的forward函數
        Args:
            x: 特徵圖，tensor shape = [batch_size, channel, height, width]
        """
        if self.with_plugins:
            # 如果有插入的層結構就會到這裡，這裡會先執行插入在conv1前的模塊
            x = self.forward_plugin(x, self.before_conv1_plugin_names)
        # 保存殘差結構的值
        residual = x

        # 通過BasicBlock的第一個conv層
        out = self.conv1(x)
        # 通過標準化層
        out = self.bn1(out)
        # 通過激活函數層
        out = self.relu(out)

        if self.with_plugins:
            # 如果有插入的層結構會在這裡，這裡會執行插入在conv1後且conv2前的層結構
            out = self.forward_plugin(out, self.after_conv1_plugin_names)

        # 通過BasicBlock的第二個conv層
        out = self.conv2(out)
        # 通過標準化層
        out = self.bn2(out)

        if self.with_plugins:
            # 如果有插入的層結構會在這裡，這裡會執行插入在conv2後且還未進行殘差結構前
            out = self.forward_plugin(out, self.after_conv2_plugin_names)

        if self.downsample is not None:
            # 如果殘差邊需要進行下採樣就會到這裡
            residual = self.downsample(x)

        # 進行相加操作
        out += residual
        # 通過激活函數層
        out = self.relu(out)

        if self.with_plugins:
            # 如果有插入的層結構會在這裡，這裡會執行最後輸出結果前差入的層結構
            out = self.forward_plugin(out, self.after_shortcut_plugin_names)

        # 最後回傳結果
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out
