# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
                         load_state_dict)
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.ops import resize
from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False):
        """
        Args:
             embed_dims: 一個特徵點會用多少維度的向量進行表示
             num_heads: 多頭注意力機制當中要用多少頭
             feedforward_channels: 在FFN時中間層的channel深度
             drop_rate: dropout的比例
             attn_drop_rate: 在attn當中的dropout率
             drop_path_rate: drop_path的機率
             num_fcs: FFN當中有多少層的全連接層，這裡預設為2
             qkv_bias: 是否啟用qkv的偏置
             act_cfg: 激活函數的使用
             norm_cfg: 標準化層的使用
             batch_first: 在使用pytorch官方的自注意模塊時可以選擇將batch_size放在最前面的維度
             attn_cfg: attn的相關設定
             ffn_cfg: FFN的相關設定
             with_cp: 是否有使用checkpoint
        """
        # 已看過ViT當中使用transformer當中的encoder部分
        super(TransformerEncoderLayer, self).__init__()

        # 構建表準化層結構，這裡將名稱以及實例化對象都接收進來
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        # 透過add_module將標準化層放到結構當中，呼叫時透過self.norm1_name進行呼叫
        self.add_module(self.norm1_name, norm1)

        # 構建attn的配置
        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                batch_first=batch_first,
                bias=qkv_bias))

        # 使用build_attn構建自注意力模塊，這裡我們將配置傳入
        self.build_attn(attn_cfg)

        # 構建自注意力結束後的標準化層，這裡也是將名稱以及實例化對象保存下來，這裡透過postfix會讓名稱與上面的標準化層有所不同
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        # 同樣使用add_module添加到結構當中
        self.add_module(self.norm2_name, norm2)

        # 構建FFN層結構的配置
        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        # 透過build_ffn構建出FFN層結構
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    def build_attn(self, attn_cfg):
        # 已看過，構建自注意力模塊並且將回傳的實例化對象直接保存下來
        self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        # 已看過，構建FFN層並且將回傳的實例化對象直接保存下來
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        # 已看過，自注意力機制的forward函數

        def _inner_forward(x):
            # 將x先進行標準化後放入到自注意力機制當中
            x = self.attn(self.norm1(x), identity=x)
            # 之後在放到FFN當中
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@BACKBONES.register_module()
class VisionTransformer(BaseModule):
    """Vision Transformer.

    This backbone is the implementation of `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at
    Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Default: True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        """
        Args:
            img_size: 輸入圖像大小，預設為224但是在大多數情況下會比預設的大上許多
            patch_size: 多少高寬會被切成一個patch，預設為16
            in_channels: 輸入的channel深度
            embed_dims: 每一個特徵點會用多少維度的向量進行表示
            num_layers: transformer的encoder要堆疊多少層，這裡預設為12
            num_heads: 在多頭注意力中多頭頭數設定，這裡預設為12
            mlp_ratio: 在FFN當中channel要加深多少倍
            out_indices: 哪層的輸出最終會做為結果輸出，預設為-1
            qkv_bias: 在qkv部分是否使用偏置
            drop_rate: dropout的比例
            attn_drop_rate: attention當中的dropout比例
            drop_path_rate: 也是dropout的一種的比例
            with_cls_token: 是否需要分類的token，因為這裡是做segmentation所以這裡默認會是False
            output_cls_token: 是否需要將分類的token進行回傳，這裡預設會是False
            norm_cfg: 設定標準化層要用哪種，這裡預設是LayerNorm
            act_cfg: 設定激活函數要用哪種，這裡預設是GELU
            patch_norm: 對於patch_embed是否需要加上一層標準化層，預設是False
            final_norm: 對於最終的特徵圖是否需要進行標準化，預設是False
            interpolate_mode: 差值方式，這裡預設是bicbic
            num_fcs: FFN當中會有多少層全連接層，這裡預設會是2
            with_cp: 是否有checkpoint
            pretrained: 預訓練相關資料
            init_cfg: 初始化的參數
        """
        # 已看過，ViT的class初始化函數
        super(VisionTransformer, self).__init__(init_cfg=init_cfg)

        if isinstance(img_size, int):
            # 當img_size是一個整數，我們透過to_2tuple進行轉換， int -> (int, int)
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            # 當img_size是一個tuple就檢查一下
            if len(img_size) == 1:
                # len=1就會需要轉換， (int) -> (int, int)
                img_size = to_2tuple(img_size[0])
            # 如果長度不是2就會直接報錯
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        if output_cls_token:
            # 當我們設定需要輸出分類token時就一定要搭配使用分類token，否則這裡產生矛盾會直接報錯
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        # 當有指定的預訓練權重時就不會有設定初始權重的方式
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            # 當我們設定中有pretrained就會跳出警告，新版本希望將pretrain資料放到init_cfg當中
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            # 將pretrained的資訊改放到init_cfg當中
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            # 其他輸入格式就會報錯
            raise TypeError('pretrained must be a str or None')

        # 保存一些相關參數
        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.pretrained = pretrained

        # self.patch_embed = 實例化PatchEmbed，也就是一開始對原始圖像進行Patch的部分
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None,
        )

        # 計算總共會有多少個patch
        num_patches = (img_size[0] // patch_size) * \
            (img_size[1] // patch_size)

        # 保存一些參數
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        # 透過nn.Parameter構建分類token，shape [1, 1, embed_dims]，這裡是可以學習的參數
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        # 構建位置編碼，這裡預先都會是0，shape [1, num_patches+1, embed_dims]
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dims))
        # dropout實例化
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            # 如果輸入的是整數表示只需要輸出其中一層的結果
            if out_indices == -1:
                # 如果給-1表示需要最後一層的輸出
                out_indices = num_layers - 1
            # 將要輸出的層保存下來，並且是用list方式進行保存
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            # 如果本身就是list或是tuple就直接保存下來
            self.out_indices = out_indices
        else:
            # 其他型態就直接報錯
            raise TypeError('out_indices must be type of int, list or tuple')

        # 構建dropout_rate，這裡比較淺層的會有比較小的dropout_rate比較深層的會有比較大的dropout_rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        # 構建layers的保存位置
        self.layers = ModuleList()
        # 遍歷總共需要多少層的encoder
        for i in range(num_layers):
            self.layers.append(
                # 將transformer的encoder部分放進去
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    batch_first=True))

        # 保存最終是否需要進行標準化
        self.final_norm = final_norm
        if final_norm:
            # 如果有需要的話就會構建標準化層
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, embed_dims, postfix=1)
            # 並且進行保存
            self.add_module(self.norm1_name, norm1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            # 當我們使用的init_weights是使用預訓練權重進行初始化就會到這裡
            # 創建一個logger進行記錄
            logger = get_root_logger()
            # checkpoint就是熟悉的dict格式，裏面的key就是對應的層結構名稱，value就是層結構的權重
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if 'pos_embed' in state_dict.keys():
                # 當預訓練權重當中有位置編碼時會進行檢查，因為位置編碼會因為輸入圖像大小不同會有不同結果
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    # 如果當前模型的位置編碼與預訓練權重位置編碼有所不同時會記錄下來，並且透過差值方式調整到相同
                    logger.info(msg=f'Resize the pos_embed shape from '
                                f'{state_dict["pos_embed"].shape} to '
                                f'{self.pos_embed.shape}')
                    # 獲取輸入圖像大小
                    h, w = self.img_size
                    # 計算出預訓練權重當中一行有多少個patch
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    # 透過resize_pos_embed進行預訓練權重的位置編碼大小調整
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)

            # 對模型進行預訓練權重加載
            load_state_dict(self, state_dict, strict=False, logger=logger)
        elif self.init_cfg is not None:
            # 如果沒有傳入就會使用父類別的初始化方式
            super(VisionTransformer, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Positiong embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        # 已看過，將圖像tensor進行位置編碼
        # patched_img = 已經經過patch後的圖像，shape [batch_size, tot_patch + 1, channel=embed_dim]
        # hw_shape = 經過patch後原始圖像新的高寬 (height, width)
        # pos_embed = 位置編碼，shape [1, tot_patch + 1, channel=embed_dim]

        # 稍微檢查patch_img與pos_embed是否合法
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            # 如果兩個不相等就要透過差值方式進行擴大，但是基本上只要配置文件沒有些錯不會有這個問題
            # 除非訓練的圖像大小不固定
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        # 這裡直接將patched_img與pos_embed直接相加，pos_embed的batch_size會自動透過廣播機制複製
        # drop_after_pos = nn.Dropout實例對象
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        # 已看過，主要是用來將預訓練的位置編碼調整到可以用到當前模型當中
        # pos_embed = 與訓練權重給的位置編碼，shape [1, total_batch, channel]
        # input_shape = 依據當前模型輸入的圖像推測出來一行總共會有多少個patch
        # pos_shape = 預訓練權重一行總共會有多少個patch
        # mode = 差值方式

        # 檢查pos_embed的維度數量
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        # 獲取預訓練位置編碼的高寬
        pos_h, pos_w = pos_shape
        # keep dim for easy deployment
        # 獲取分類token的位置編碼，shape [1, 1, channel]
        cls_token_weight = pos_embed[:, 0:1]
        # 獲取除了分類token以外的位置編碼，shape [1, pos_h * pos_w, channel]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        # [1, pos_h * pos_w, channel] -> [1, pos_h, pos_w, channel] -> [1, channel, pos_h, pos_w]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        # 透過resize將高寬調整到input_shape，這裡是透過官方的差值方式進行縮放
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        # [1, channel, height, width] -> [1, channel, height * width] -> [1, height * width, channel]
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        # 將分類token拼接上去，shape [1, height * width + 1, channel]
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        # 最後回傳調整好的位置編碼
        return pos_embed

    def forward(self, inputs):
        # 已看過，完整ViT模型的forward函數
        # inputs = 輸入的訓練圖像，shape [batch_size, channel, height, width]

        # 獲取batch_size
        B = inputs.shape[0]

        # 將input放入到patch_embed當中進行向前傳遞
        # x = 經過patch後的tensor，shape [batch_size, height * width, channel=embed_dim]
        # hw_shape = 經過patch後原始圖像新的高寬(用來之後可以將x轉回成2d特徵圖)
        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        # 使用廣播機制對分類token進行擴圍，shape [1, height * width, channel] -> [batch_size, height * width, channel]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # 將類別token進行拼接，[batch_size, tot_patch, channel] -> [batch_size, tot_patch + 1, channel]
        x = torch.cat((cls_tokens, x), dim=1)
        # 透過_pos_embedding將x添加上位置編碼，shape不會產生變化
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            # 如果我們設定不需要類別token就會在這裡去除
            # x shape [batch_size, tot_patch + 1, channel] -> [batch_size, tot_patch, channel]
            x = x[:, 1:]

        # 記錄下每層encoder的輸出
        outs = []
        # 遍歷每層encoder
        for i, layer in enumerate(self.layers):
            # 將x放入到自注意力機制進行向前傳播，同時x的shape不會改變
            x = layer(x)
            if i == len(self.layers) - 1:
                # 如果已經到最後一層就看是否需要通過標準化層
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                # 如果該層是指定的輸出層就會進來
                if self.with_cls_token:
                    # 如果我們需要類別token還是需要先將類別token移除
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                # 獲取當前的shape
                B, _, C = out.shape
                # 將1d特徵圖轉成2d的，[batch_size, tot_patch, channel] -> [batch_size, height, width, channel]
                # -> [batch_size, channel, height, width]
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    # 如果需要類別token就會放到out當中
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        super(VisionTransformer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
