# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        """
        :param backbone: 骨幹部分的配置內容，這裡還會有一層也就是會有type在裡面，type指定的就是backbone類型
        :param decode_head: 使用的解碼頭，這裡還會有一層也就是會有type在裡面，type決定要用哪種解碼頭
        :param neck: 如果從backbone出來的特徵圖還有需要進行加工就會有這一層
        :param auxiliary_head: 輔助訓練部分，這裡還會有一層也就是會有type在裡面，type決定要用哪種輔助分類頭
        :param train_cfg: 訓練的設定檔
        :param test_cfg: 測試的設定檔，這裡會寫是要用whole模式或是slide模式進行測試
        :param pretrained: 使用的預訓練權重下載位置
        :param init_cfg: 用來控制初始化用的，預設為None
        """
        # 已看過，空下來的部分表示還不清楚

        # 初始化繼承的class，最後會初始化到最底層的module class
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            # 如果有設定pretrained那麼backbone中的pretrained就必須要關閉
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            # 將backbone的pretrained設定成跟EncoderDecoder當中的pretrained一樣
            backbone.pretrained = pretrained
        # 透過builder中的build_backbone構建backbone，同時將backbone設定資料傳入
        # self.backbone = 構建好的backbone，這裡就是backbone的實例對象了裡面包括了encoder
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            # 如果有neck結構就會進行neck結構的構建，這裡我們先不去看，因為還沒有遇到
            # neck也會是一系列層結構，主要會是在如果需要多層backbone輸出時會使用到，作為中間處理這些輸出層用的
            self.neck = builder.build_neck(neck)
        # decode_head = 解碼頭的配置參數，使用backbone特徵提取出來的特徵圖進行最後的預測
        # _init_decode_head會構建出完整解碼頭
        self._init_decode_head(decode_head)
        # auxiliary_head = 輔助訓練頭，使用backbone特徵提取出來的特徵圖進行預測，在訓練過程中可以讓淺層網路有比較好的學習效果
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 檢查一定要設定解碼頭，如果沒有半個解碼頭這裡就會報錯
        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        # 已看過
        # 初始化解碼頭，decode_head = 解碼頭的config內容

        # 透過build_head構建解碼頭實例對象
        # self.decode_head = 解碼頭的實例對象，也會有損失計算在內
        self.decode_head = builder.build_head(decode_head)
        # self.align_corners = 將decode_head中的align_corners拿出來
        self.align_corners = self.decode_head.align_corners
        # self.num_classes = 分類類別數
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        # 已看過
        # 初始化輔助解碼頭，auxiliary_head = 輔助解碼頭配置參數

        # 如果傳入的不是None就會開始配置
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                # 如果是list型態就會透過nn.ModuleList變成列表
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                # 如果是dict就直接開始解析dict，這裡也是調用build_head構建，所以其實跟decode_head相同
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        # 已看過，主要是用來從圖像提取出特徵圖
        # img shape [batch_size, channel, height, width]
        # 將原始tensor傳入到backbone當中進行特徵提取
        # x = list[tensor]，tensor shape [batch_size, channel, height, width]，list長度就是有多少輸出出來的特徵層
        # x的最後一個就是decoder最終的輸出
        # 如果backbone是transformer型態x的輸出也會是2d的
        x = self.backbone(img)
        if self.with_neck:
            # 如果有neck層結構就會進入到neck當中進行正向傳播
            x = self.neck(x)
        # 回傳特徵層
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # 已看過，進行encode_decode過成
        # img = 圖像的tensor，shape [batch_size, channel, height, width]
        # img_metas = 圖像的相關內容

        # 將輸入的圖像放入到extract_feat進行特徵提取，這裡與train相同都會經過encoder和decoder
        # x shape [batch_size, channel, height, width]
        x = self.extract_feat(img)
        # 再將decoder的輸出放到解碼頭當中，out shape [batch_size, channel, height, width]
        out = self._decode_head_forward_test(x, img_metas)
        # 最後將結果透過resize進行大小調整，將out的圖像高寬調整到與輸入時相同
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # 最後將結果回傳
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in training."""
        # 已看過
        # 將特徵圖透過解碼頭預測出最終結果，並且與正確圖像進行損失計算
        # 創建loss字典
        losses = dict()
        # 透過decode_head中的forward_train進行
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        # 在key前面加上decode
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        # 已看過，調用test模式的解碼頭
        # 這裡有將test的config傳入
        # seg_logits = shape [batch_size, channel, height, width]
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        # 已看過，如果有輔助訓練就會到這裡來
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            # 如果是ModuleList就到這裡進行遍歷
            for idx, aux_head in enumerate(self.auxiliary_head):
                # 調用該類別的forward_train函數
                # loss_aux = 計算出輔助訓練的loss值
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                # 將輔助函數的loss更新到losses當中，這裡會透過idx對key的名稱進行改變
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        # 將最後獲得的losses進行回傳
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        """
        :param img: 一個batch圖像的tensor堆疊，shape [batch_size, channel, height, width] 
        :param img_metas: list裏面的資料是每張圖像的詳細資訊，list長度會與batch_size相同
        :param gt_semantic_seg: 一個batch的標註圖像tensor堆疊，shape [batch_size, channel, height, width]
        :return: 一個由dict組成的loss
        """
        # 已看過

        # x = list或是tuple[tensor]，tensor shape [batch_size, channel, height, width]，list長度就是有多少輸出出來的特徵層
        # x最後一層的輸出就是encoder的輸出(如果沒有neck層結構的話)
        x = self.extract_feat(img)

        # 構建losses字典，這個會是最後要回傳的損失計算值
        losses = dict()

        # 透過decode_head_forward_train將特徵圖變成最終的預測圖，同時傳入標註圖像進行損失計算
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        # 將loss_decode的值更新到losses當中
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            # 如果有使用輔助分類頭就會到這裡計算輔助分類損失
            # 這裡與上面的decode_head_forward_train幾乎相同，只是差在解碼的地方有些不同而以
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            # 將輔助訓練部分獲得的loss字典更新上去
            losses.update(loss_aux)

        # 最後將整個損失計算的東西返回出去
        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        # 已看過，使用滑動窗口方式進行驗證

        # 獲取滑動窗口的步距
        h_stride, w_stride = self.test_cfg.stride
        # 獲取裁切大小
        h_crop, w_crop = self.test_cfg.crop_size
        # 獲取當前輸入圖像的相關資訊
        batch_size, _, h_img, w_img = img.size()
        # 獲取分類類別數
        num_classes = self.num_classes
        # 目前暫時不知道h_grids與w_grids的作用
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        # 構建出預測的tensor，shape [batch_size, num_classes, height, width]
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        # 計數用的tensor，shape [batch_size, 1, height, width]
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        # 進行遍歷h_grids長度以及w_grids長度
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                # 獲取當前需要的圖像左上角右下角點
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                # 根據左上角以及右下角對原始圖像進行截取
                # crop_img shape = [batch_size, channel, height=y2-y1, width=x2-x1]
                crop_img = img[:, :, y1:y2, x1:x2]
                # 將裁切下來的圖像放入到encoder_decode的forward當中
                # crop_seg_logit shape 會與輸入時完全相同，因為在最後面會透過resize將模型輸出的圖像進行調整
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                # 透過F.pad將crop_seg_logit進行padding並且全部為0，這樣就可以跟preds進行相加
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                # 將有計算到的部分加一，應該最後每個點都會進行取平均
                count_mat[:, :, y1:y2, x1:x2] += 1
        # 理論上來說經過滑動窗口後每一個點都會被輸入到模型當中進行預測，所以不會有0的地方
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        # 這裡會根據每個點被計算到的次數進行曲平均
        preds = preds / count_mat
        if rescale:
            # remove padding area，進行重新調整大小
            # 獲取輸入時的圖像大小
            resize_shape = img_meta[0]['img_shape'][:2]
            # 將preds重新調整大小到與輸入時相同，正常來說這裡不會有任何改變
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                # remove padding area
                resize_shape = img_meta[0]['img_shape'][:2]
                seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        # 已看過，進行測試的正向推理
        # 這裡測試會有兩種模式一種會是滑動另一種是整張
        assert self.test_cfg.mode in ['slide', 'whole']
        # 獲取原始圖像大小
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            # 如果驗證模式是slide就會到這裡
            # seg_logit shape = [batch_size, channel, height, width]，這裡的高寬會是圖像最原始的高寬，不會是透過數據流的高寬
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            # 如果驗證模式是whole就會到這裡
            seg_logit = self.whole_inference(img, img_meta, rescale)
        # 透過softmax獲取分類類別概率
        output = F.softmax(seg_logit, dim=1)
        # 獲取是否有經過翻轉
        flip = img_meta[0]['flip']
        if flip:
            # 如果有經過翻轉，我們就要看翻轉方式
            flip_direction = img_meta[0]['flip_direction']
            # 這裡會檢查一下
            assert flip_direction in ['horizontal', 'vertical']
            # 根據翻轉方式將圖像轉回來
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        # 最後輸出
        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        # 已看過，這裡就是簡單的進行測試，在這裡只會處理一張圖像的預測
        # img = 圖像的tensor格式，shape [batch_size=1, channel=3, height, width]
        # img_meta = 當前圖像的相關資料
        # rescale = 好像是會調整大小，預設為True

        # 透過infernce進行預測
        # seg_logit shape = [batch_size, channel, height, width]，這裡的高寬會是最原始圖像的高寬
        seg_logit = self.inference(img, img_meta, rescale)
        # 透過argmax獲取該像素點上概率最高的分類類別，seg_pred shape [batch_size, height, width]
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        # 將seg_pred從tensor格式轉成numpy格式
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim，轉成list
        seg_pred = list(seg_pred)
        # 回傳
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
