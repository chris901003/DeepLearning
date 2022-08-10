# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmocr.models.builder import (RECOGNIZERS, build_backbone, build_convertor,
                                  build_decoder, build_encoder, build_loss,
                                  build_preprocessor)
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class EncodeDecodeRecognizer(BaseRecognizer):
    """Base class for encode-decode recognizer."""

    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 max_seq_len=40,
                 pretrained=None,
                 init_cfg=None):
        """ 已看過，文字判讀的encoder與decoder部分
        Args:
            preprocessor: 預處理設定
            backbone: 骨幹設定config資料
            encoder: encoder設定資料
            decoder: decoder設定資料
            loss: 損失計算構建方式
            label_convertor: 將預測出來的結果轉成損失計算的資訊(CTC)
            train_cfg: train時資料的處理
            test_cfg: test時資料的處理
            max_seq_len: 最大序列長度
            pretrained: 預訓練權重資料
            init_cfg: 初始化方式設定
        """

        # 繼承自BaseRecognizer，對繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)

        # Label convertor (str2tensor, tensor2str)
        # 在文字判讀部分一定會需要label的轉換器
        assert label_convertor is not None
        # 將最大序列長度傳入進去
        label_convertor.update(max_seq_len=max_seq_len)
        # 構建label轉換實例對象
        self.label_convertor = build_convertor(label_convertor)

        # Preprocessor module, e.g., TPS
        # 先將預處理部分設定成None
        self.preprocessor = None
        if preprocessor is not None:
            # 如果有設定後處理部分就會到這裡，對後處理方式進行實例化
            self.preprocessor = build_preprocessor(preprocessor)

        # Backbone
        # 檢查是否傳入backbone參數
        assert backbone is not None
        # 構建backbone部分
        self.backbone = build_backbone(backbone)

        # Encoder module
        # 先將encoder部分設定成None
        self.encoder = None
        if encoder is not None:
            # 如果有設定encoder部分就會到這裡進行初始化設定，並且獲取encoder的實例對象
            self.encoder = build_encoder(encoder)

        # Decoder module
        # 檢查是否有傳入decoder設定資料，這裡我們一定需要構建decoder
        assert decoder is not None
        # 透過label_convertor獲取總共需要多少分類
        decoder.update(num_classes=self.label_convertor.num_classes())
        # 透過label_convertor獲取開始的index值
        decoder.update(start_idx=self.label_convertor.start_idx)
        # 透過label_convertor獲取padding的index值
        decoder.update(padding_idx=self.label_convertor.padding_idx)
        # 透過label_convertor獲取最大序列長度
        decoder.update(max_seq_len=max_seq_len)
        # 最後進行構建decoder實例對象
        self.decoder = build_decoder(decoder)

        # Loss
        # 檢查是否傳入損失計算方式，這裡一定需要傳入損失計算方式
        assert loss is not None
        # 將padding_idx傳入到loss當中
        loss.update(ignore_index=self.label_convertor.padding_idx)
        # 構建損失實例對象
        self.loss = build_loss(loss)

        # 保存一些傳入的參數
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.max_seq_len = max_seq_len

        if pretrained is not None:
            # 如果有傳入pretrained就會需要將pretrained放到init_cfg當中
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone."""
        # 已看過，進行特徵提取的forward函數
        if self.preprocessor is not None:
            # 如果有需要進行預處理會到這裡進行
            img = self.preprocessor(img)

        # 進行特徵提取，x shape [batch_size, channel, height, width]
        x = self.backbone(img)

        # 將提取的特徵圖回傳
        return x

    def forward_train(self, img, img_metas):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'filename', and may also contain
                'ori_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        # 已看過，文字判讀開始訓練的forward函數
        # img = 一個batch的圖像資料，tensor shape [batch_size, channel, height, width]
        # img_metas = 一個batch的圖像詳細資料

        # 遍歷img_metas當中的資料，也就是遍歷每張圖像的資料
        for img_meta in img_metas:
            # 獲取valid_ratio值
            valid_ratio = 1.0 * img_meta['resize_shape'][1] / img.size(-1)
            # 將資料放到img_meta當中保存
            img_meta['valid_ratio'] = valid_ratio

        # 進行特徵提取
        feat = self.extract_feat(img)

        # 將標註資訊讀取出來
        gt_labels = [img_meta['text'] for img_meta in img_metas]

        # targets_dict = dict{
        #   'targets': 原始標註資訊轉成文字對應index後的tensor格式，list[tensor]且tensor shape [len_gt_words]
        #   'flatten_target': 將標註的index全部展平，tensor shape [total_words]
        #   'target_lengths': 每一張圖像的標註文字長度，tensor shape [batch_size]
        # }
        targets_dict = self.label_convertor.str2tensor(gt_labels)

        out_enc = None
        if self.encoder is not None:
            # 如果有encoder就會到這裡進行向前傳播
            out_enc = self.encoder(feat, img_metas)

        # 進行decoder部分的向前傳播
        # out_dec shape = [batch_size, width, channel]
        out_dec = self.decoder(
            feat, out_enc, targets_dict, img_metas, train_mode=True)

        # 構建loss需要的資料
        loss_inputs = (
            out_dec,
            targets_dict,
            img_metas,
        )
        # 進行損失計算
        losses = self.loss(*loss_inputs)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            img (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        # 已看過，這裡會對單張圖像進行預測
        # img = 要預測的圖像，tensor shape [batch_size=1, channel, height, width]
        # img_metas = 該圖像的詳細資訊
        # kwargs = 會有rescale，是否需要將圖像進行resize

        for img_meta in img_metas:
            # 獲取valid_ratio資訊
            valid_ratio = 1.0 * img_meta['resize_shape'][1] / img.size(-1)
            img_meta['valid_ratio'] = valid_ratio

        # 進行特徵提取，feat = tensor shape [batch_size=1, channel, height, width]
        feat = self.extract_feat(img)

        out_enc = None
        if self.encoder is not None:
            # 如果有encoder層結構就會通過
            out_enc = self.encoder(feat, img_metas)

        # 通過decoder層結構，獲取預測結果
        out_dec = self.decoder(
            feat, out_enc, None, img_metas, train_mode=False)

        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return out_dec

        # label_indexes = list[list]，第一個list長度會是batch_size，第二個list長度會是該圖像最終字串長度，這裡會是預測的index
        # label_scores = 型態與label_indexes相同，只是裡面存的資料是置信度
        label_indexes, label_scores = self.label_convertor.tensor2idx(
            out_dec, img_metas)
        # 將index轉成最後的字串，label_strings = list[str]，list長度就會是batch_size
        label_strings = self.label_convertor.idx2str(label_indexes)

        # flatten batch results
        # 最終回傳結果
        results = []
        for string, score in zip(label_strings, label_scores):
            # 將預測的字串與每個文字的置信度包成dict後放到results當中
            results.append(dict(text=string, score=score))

        return results

    def merge_aug_results(self, aug_results):
        # 已看過，合併results資料
        # 先將out_text與out_score初始化
        out_text, out_score = '', -1
        # 遍歷aug_results當中資料，也就是遍歷一個batch的資料
        for result in aug_results:
            # 獲取預測的文字
            text = result[0]['text']
            # 這裡會求出平均置信度分數
            score = sum(result[0]['score']) / max(1, len(text))
            if score > out_score:
                # 如果平均分數大於閾值就會到這裡
                out_text = text
                out_score = score
        # 最終構成輸出的results
        out_results = [dict(text=out_text, score=out_score)]
        return out_results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function as well as time augmentation.

        Args:
            imgs (list[tensor]): Tensor should have shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): The metadata of images.
        """
        # 已看過，測試部分調用
        # imgs = 輸入到網路的圖像資料，list[tensor]，list長度會是要檢測的圖像數量
        #        tensor shape [batch_size=1, channel, height, width]
        # img_metas = 圖像的詳細資料

        # 最終結果保存的地方
        aug_results = []
        # 遍歷需要預測的圖像
        for img, img_meta in zip(imgs, img_metas):
            # 透過simple_test進行預測並且獲取結果
            # result = list[dict]
            # dict = {
            #   'text': 預測出來的字串
            #   'score': list[float]，list長度會與text長度相同，每個代表對應index的置信度分數
            # }
            result = self.simple_test(img, img_meta, **kwargs)
            # 將結果保存
            aug_results.append(result)

        # 將結果進行合併後回傳
        return self.merge_aug_results(aug_results)
