# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..skeleton_gcn.utils import Graph


def zero(x):
    """return zero."""
    return 0


def identity(x):
    """return input itself."""
    return x


class STGCNBlock(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph
    sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and
            graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out},
            V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V,
            V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        """ STGCN的模塊，這裡會進行時空圖卷積
        Args:
            in_channels: 輸入的channel深度
            out_channels: 輸出的channel深度
            kernel_size: 卷積核大小(時間維度卷積核, 空間維度卷積核)
            stride: 步距
            dropout: dropout率
            residual: 是否使用殘差模塊
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()

        # 檢查kernel_size當中有指定時間與空間的卷積核大小
        assert len(kernel_size) == 2
        # 這裡會限制時間方面的卷積核需要是奇數
        assert kernel_size[0] % 2 == 1
        # 創建padding大小
        padding = ((kernel_size[0] - 1) // 2, 0)

        # 構建GCN卷積模塊，這裡是對於空間維度進行卷積
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        # 構建對於時間維度卷積的實例化對象，透過nn.Sequential將一系列操作包裝
        self.tcn = nn.Sequential(
            # 會先將gcn結果進行標準化以及激活
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            # 透過conv2d進行卷積，這裡使用的就會是9x1的卷積核
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            # 將時間維度卷積結果通過標準化以及激活函數
            nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True))

        if not residual:
            # 如果沒有使用殘差結構就設定成zero
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            # 如果輸入的channel深度與輸出的channel深度相同且步距為1就可以直接使用殘差值
            self.residual = identity

        else:
            # 否則就需要透過conv進行調整
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    # 調整輸出channel深度
                    out_channels,
                    kernel_size=1,
                    # 調整步距
                    stride=(stride, 1)), nn.BatchNorm2d(out_channels))

        # 最後實例化一個激活函數
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj_mat):
        """Defines the computation performed at every call."""
        # 進行STGCNBlock的forward函數
        # x = 關節點資訊，shape [batch_size * people, channel, frames, num_node]
        # adj_mat = 鄰接矩陣資訊，tensor shape [3, num_node, num_node]，這裡的3是根據論文的分類方式會有3群

        # 將x先通過residual模塊，如果沒有要使用residual就會是0
        res = self.residual(x)
        # 進行主幹forward，會先通過gcn模塊進行空間卷積，x shape [batch_size * people, channel, frame, num_point]
        x, adj_mat = self.gcn(x, adj_mat)
        # 之後在通過tcn進行時間卷積，最後與殘差相加
        x = self.tcn(x) + res

        # 回傳激活過後的結果以及鄰接矩陣
        return self.relu(x), adj_mat


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution.
            Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides
            of the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}
            , V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)
            ` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        """ 最基礎的圖卷積模塊初始化部分，這裡是對於空間維度的卷積
        Args:
            in_channels: 輸入的channel深度
            out_channels: 輸出的channel深度
            kernel_size: 空間卷積核大小
            t_kernel_size: 時間方向的卷積核大小
            t_stride: 時間方向的步距
            t_padding: 時間方向的padding
            t_dilation: 時間方向的膨脹係數
            bias: 是否使用偏置
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()

        # 將空間方向的卷積核大小保存
        self.kernel_size = kernel_size
        # 使用conv2d進行卷積
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            # 這裡的kernel大小會是1x1與論文當中的3x1有不同，但是不是錯的到下面forward當中進行解釋
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, adj_mat):
        """Defines the computation performed at every call."""
        # 進行空間方向的卷積
        # x = 關節點資訊，shape [batch_size * people, channel, frames, num_node]
        # adj_mat = 鄰接矩陣資料，shape [3, num_node, num_node]

        # 檢查鄰接矩陣的類別數量要與kernel大小相同
        assert adj_mat.size(0) == self.kernel_size

        # 進行卷積
        x = self.conv(x)

        # 獲取x的shape [batch_size * people, kernel_size * out_channel, frames, num_node]
        n, kc, t, v = x.size()
        # 進行通道調整 [batch_size, kernel_size, out_channel, frames, num_node]
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        # 透過einsum進行鄰接矩陣計算，x shape [batch_size * num_people, channel, frames, num_node]
        # 這裡會沿著k也就是kernel部分進行矩陣乘法，這樣就是分組部分的特徵計算
        x = torch.einsum('nkctv,kvw->nctw', (x, adj_mat))

        # 將結果在記憶體中連續結果回傳以及鄰接矩陣回傳
        return x.contiguous(), adj_mat


@BACKBONES.register_module()
class STGCN(nn.Module):
    """Backbone of Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data.
        graph_cfg (dict): The arguments for building the graph.
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph. Default: True.
        data_bn (bool): If 'True', adds data normalization to the inputs.
            Default: True.
        pretrained (str | None): Name of pretrained model.
        **kwargs (optional): Other parameters for graph convolution units.

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 in_channels,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 pretrained=None,
                 **kwargs):
        """ STGCN初始化函數
        Args:
            in_channels: 輸入channel深度
            graph_cfg: 構建圖表的數據格式
            edge_importance_weighting: 如果設定為True就會加上邊緣的可學習權重
            data_bn: 如果為True就會將輸入進行標準化
            pretrained: 預訓練權重資料
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()

        # load graph
        # 構建Graph實例化對象，將graph_cfg傳入進行初始化，主要是在構建關節點之間的連線，最後形成的鄰接矩陣表
        self.graph = Graph(**graph_cfg)
        # 將graph當中的A轉成tensor格式，A保存的是鄰接矩陣資料，A的shape會因為不同設定有所不同[_, num_node, num_node]
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # 將A註冊到register_buffer當中並且名稱為'A'
        self.register_buffer('A', A)

        # build networks
        # 獲取空間維度上的卷積核大小
        spatial_kernel_size = A.size(0)
        # 獲取時空維度上的卷積核大小
        temporal_kernel_size = 9
        # 將時間以及空間維度卷積打包
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        # 如果有設定使用標準化層就會構建，channel深度會是輸入的channel乘上關節點數量
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else identity

        # 遍歷kwargs當中的內容，如果有的話就會放到kwargs0當中
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        # 構建一系列STGCN模塊，會放到nn.ModuleList當中，堆疊10層STGCN的Block結構
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, kernel_size, 1, **kwargs),
            STGCNBlock(64, 128, kernel_size, 2, **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, **kwargs),
            STGCNBlock(128, 128, kernel_size, 1, **kwargs),
            STGCNBlock(128, 256, kernel_size, 2, **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, **kwargs),
            STGCNBlock(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            # 如果有需要設定可學習的邊重要性權重就會到這裡
            self.edge_importance = nn.ParameterList([
                # 這裡會構建的數量與STGCNBlock堆疊數量相同且shape與A相同，預設會都是1
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            # 否則就會是固定的1，不可學習的值
            self.edge_importance = [1 for _ in self.st_gcn_networks]

        # 將pretrained保存
        self.pretrained = pretrained

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        # data normalization
        # 進行特徵提取的forward函數
        # x = tensor shape [batch_size, channel(x, y, score), frames, num_node, people]

        # 將x轉成float格式
        x = x.float()
        # 獲取x的shape資訊，這裡可以與上面進行對照
        n, c, t, v, m = x.size()  # bs 3 300 25(17) 2
        # 將通道進行調整 [batch_size, people, num_node, channel, frames]
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N M V C T
        # 將通道進行調整 [batch_size * people, num_node * channel, frames]
        x = x.view(n * m, v * c, t)
        # 將x通過標準化層結構，這裡會對num_node * channel維度進行標準化
        x = self.data_bn(x)
        # 將通道進行調整 [batch_size, people, num_node, channel, frames]
        x = x.view(n, m, v, c, t)
        # 將通道進行調整 [batch_size, people, channel, frames, num_node]
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        # 將通道進行調整 [batch_size * people, channel, frames, num_node]
        # 透過通道調整後就會與在做2D的CNN卷積時有類似的shape將frames作為height將num_node作為width
        x = x.view(n * m, c, t, v)  # bsx2 3 300 25(17)

        # forward
        # 將x通過多層STGCNBlock層結構
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        return x
