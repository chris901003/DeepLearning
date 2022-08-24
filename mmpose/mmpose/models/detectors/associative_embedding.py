# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import torch
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from mmpose.core.evaluation import (aggregate_scale, aggregate_stage_flip,
                                    flip_feature_maps, get_group_preds,
                                    split_ae_outputs)
from mmpose.core.post_processing.group import HeatmapParser
from mmpose.core.visualization import imshow_keypoints
from .. import builder
from ..builder import POSENETS
from .base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class AssociativeEmbedding(BasePose):
    """Associative embedding pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            ``loss_keypoint`` for heads instead.
    """

    def __init__(self,
                 backbone,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        """ 關聯嵌入關節點檢測器初始化函數
        Args:
            backbone: 特徵提取模塊設定資料
            keypoint_head: 關節點解碼頭設定資料
            train_cfg: 訓練時對於資料的調整
            test_cfg: 測試時對於資料的調整
            pretrained: 預訓練權重地址，可以是檔案位置也可以是網址
            loss_pose: 不推薦使用這個參數，所以通常會是空
        """
        # 繼承自BasePose，對繼承對象進行初始化
        # BasePose沒有init函數，而BasePose繼承自nn.Module，所以這裡會直接到nn.Module的init函數
        super().__init__()
        # 將單精度訓練模式設定成False，這裡就會直接使用原始的雙精度進行訓練
        self.fp16_enabled = False

        # 構建backbone模塊
        self.backbone = builder.build_backbone(backbone)

        if keypoint_head is not None:
            # 如果有設定關節點解碼頭就會到這裡
            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                # 如果解碼頭設定檔當中沒有loss_keypoint就會跳出警告，新版本需要將loss_pose直接放到keypoint_head當中
                # 不是透過初始化函數調用時傳入
                warnings.warn(
                    '`loss_pose` for BottomUp is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                # 將loss_keypoint資料傳入到kepoint_head當中
                keypoint_head['loss_keypoint'] = loss_pose

            # 構建關節點解碼頭實例對象
            self.keypoint_head = builder.build_head(keypoint_head)

        # 將傳入資料進行保存
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # 看是否需要使用upd，這個晚點研究
        self.use_udp = test_cfg.get('use_udp', False)
        # 構建熱力圖資料處理實例化對象
        self.parser = HeatmapParser(self.test_cfg)
        # 將預訓練權重放入
        self.pretrained = pretrained
        # 進行初始化權重
        self.init_weights()

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        if self.with_keypoint:
            self.keypoint_head.init_weights()

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img=None,
                targets=None,
                masks=None,
                joints=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss is True.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_img_channel: C
            - img_width: imgW
            - img_height: imgH
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M

        Args:
            img (torch.Tensor[N,C,imgH,imgW]): Input image.
            targets (list(torch.Tensor[N,K,H,W])): Multi-scale target heatmaps.
            masks (list(torch.Tensor[N,H,W])): Masks of multi-scale target
                heatmaps
            joints (list(torch.Tensor[N,M,K,2])): Joints of multi-scale target
                heatmaps for ae loss
            img_metas (dict): Information about val & test.
                By default it includes:

                - "image_file": image path
                - "aug_data": input
                - "test_scale_factor": test scale factor
                - "base_size": base size of input
                - "center": center of image
                - "scale": scale of image
                - "flip_index": flip index of keypoints
            return loss (bool): ``return_loss=True`` for training,
                ``return_loss=False`` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if 'return_loss' is true, then return losses. \
                Otherwise, return predicted poses, scores, image \
                paths and heatmaps.
        """

        if return_loss:
            # 如果需要計算損失會到這裡
            return self.forward_train(img, targets, masks, joints, img_metas, **kwargs)
        # 如果不需要計算損失會到這裡
        return self.forward_test(img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, targets, masks, joints, img_metas, **kwargs):
        """Forward the bottom-up model and calculate the loss.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M

        Args:
            img (torch.Tensor[N,C,imgH,imgW]): Input image.
            targets (List(torch.Tensor[N,K,H,W])): Multi-scale target heatmaps.
            masks (List(torch.Tensor[N,H,W])): Masks of multi-scale target heatmaps
            joints (List(torch.Tensor[N,M,K,2])): Joints of multi-scale target heatmaps for ae loss
            img_metas (dict):Information about val&test
                By default this includes:
                - "image_file": image path
                - "aug_data": input
                - "test_scale_factor": test scale factor
                - "base_size": base size of input
                - "center": center of image
                - "scale": scale of image
                - "flip_index": flip index of keypoints

        Returns:
            dict: The total loss for bottom-up
        """
        """ 進行向前傳遞
        Args:
            img: 圖像資料，tensor shape [batch_size, channel, height, width]
            targets: 標註熱力圖圖像，list[tensor]且tensor shape [batch_size, num_joints, height, width]
            masks: 標註哪些部分不需要計算loss，list[tensor]且tensor shape [batch_size, height, width]
            joints: 關節點位置，list[tensor]且tensor shape [batch_size, max_people_pre_picture, num_joints, 2]
            img_metas: 圖像的詳細資訊
            kwargs: 其他資訊
        """

        # 獲取圖像特徵，透過backbone提取特徵，output = tensor shape [batch_size, channel, height, width]
        output = self.backbone(img)

        if self.with_keypoint:
            # 如果有關節點解碼頭就會到這裡，output = list[tensor]，tensor shape [batch_size, channel, height, width]
            output = self.keypoint_head(output)

        # if return loss
        # 構建計算損失的字典
        losses = dict()
        if self.with_keypoint:
            # 計算預測損失值
            keypoint_losses = self.keypoint_head.get_loss(output, targets, masks, joints)
            # 更新losses字典
            losses.update(keypoint_losses)

        # 將losses回傳
        return losses

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Outputs.
        """
        output = self.backbone(img)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Inference the bottom-up model.

        Note:
            - Batchsize: N (currently support batchsize = 1)
            - num_img_channel: C
            - img_width: imgW
            - img_height: imgH

        Args:
            flip_index (List(int)):
            aug_data (List(Tensor[NxCximgHximgW])): Multi-scale image
            test_scale_factor (List(float)): Multi-scale factor
            base_size (Tuple(int)): Base size of image when scale is 1
            center (np.ndarray): center of image
            scale (np.ndarray): the scale of image
        """
        """ 如果是測試模式就會到這裡進行正向傳遞
        Args:
            img: 原始圖像資料，tensor shape [batch_size, height, width, channel]
            img_metas: 圖像詳細資料
            return_heatmap: 是否需要回傳熱力圖
            kwargs: 其他參數
        """
        # 這裡只允許單一圖像進行測試，所以batch_size部分需要是1
        assert img.size(0) == 1
        # 同時img_metas也要是1
        assert len(img_metas) == 1

        # 獲取圖像詳細資料
        img_metas = img_metas[0]

        # 獲取圖像增強的資料
        aug_data = img_metas['aug_data']

        # 獲取img_metas當中資訊
        test_scale_factor = img_metas['test_scale_factor']
        base_size = img_metas['base_size']
        center = img_metas['center']
        scale = img_metas['scale']

        # 最終結果保存地方
        result = {}

        # 獲取熱力圖list保存位置
        scale_heatmaps_list = []
        scale_tags_list = []

        # 遍歷縮放倍率
        for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
            # 獲取aud_data當中圖像資料，這裡會將圖像放到測試設備上
            image_resized = aug_data[idx].to(img.device)

            # 進行特徵提取，features shape = [batch_size, channel, height, width]
            features = self.backbone(image_resized)
            if self.with_keypoint:
                # 如果有關節點解碼頭就會到這裡，outputs = list[tensor]，tensor shape [batch_size, channel, height, width]
                outputs = self.keypoint_head(features)

            # 將tags以及熱力圖資料提取出來
            heatmaps, tags = split_ae_outputs(
                outputs, self.test_cfg['num_joints'],
                self.test_cfg['with_heatmaps'], self.test_cfg['with_ae'],
                self.test_cfg.get('select_output_index', range(len(outputs))))

            if self.test_cfg.get('flip_test', True):
                # 如果有需要flip_test就會到這裡，會將圖像進行左右翻轉後再次放到模型當中預測
                # use flip test
                # 透過torch.flip將圖像進行翻轉，之後放到backbone當中提取特徵
                features_flipped = self.backbone(torch.flip(image_resized, [3]))
                if self.with_keypoint:
                    # 如果有關節點分類頭也需要進行傳遞
                    outputs_flipped = self.keypoint_head(features_flipped)

                # 將翻轉後的圖像資料進行熱力圖以及tags的提取
                heatmaps_flipped, tags_flipped = split_ae_outputs(
                    outputs_flipped, self.test_cfg['num_joints'],
                    self.test_cfg['with_heatmaps'], self.test_cfg['with_ae'],
                    self.test_cfg.get('select_output_index', range(len(outputs))))

                # 將熱力圖進行翻轉，這樣就可以翻轉回來
                heatmaps_flipped = flip_feature_maps(heatmaps_flipped, flip_index=img_metas['flip_index'])
                if self.test_cfg['tag_per_joint']:
                    # 如果有設定tag_per_joint就會到這裡
                    tags_flipped = flip_feature_maps(tags_flipped, flip_index=img_metas['flip_index'])
                else:
                    # 沒有設定tag_per_joint就會到這裡
                    tags_flipped = flip_feature_maps(tags_flipped, flip_index=None, flip_output=True)

            else:
                # 如果沒有需要將圖像進行左右變換後再次驗證就會到這裡，將flipped的資料設定成None
                heatmaps_flipped = None
                tags_flipped = None

            # 將熱力圖翻轉的結果與原始圖像預測結果進行融合
            aggregated_heatmaps = aggregate_stage_flip(
                heatmaps,
                heatmaps_flipped,
                index=-1,
                project2image=self.test_cfg['project2image'],
                size_projected=base_size,
                align_corners=self.test_cfg.get('align_corners', True),
                aggregate_stage='average',
                aggregate_flip='average')

            # 將tags翻轉的結果與原始圖像預測結果進行融合
            aggregated_tags = aggregate_stage_flip(
                tags,
                tags_flipped,
                index=-1,
                project2image=self.test_cfg['project2image'],
                size_projected=base_size,
                align_corners=self.test_cfg.get('align_corners', True),
                aggregate_stage='concat',
                aggregate_flip='concat')

            if s == 1 or len(test_scale_factor) == 1:
                # 如果s==1或是test_scale_factor只有1個就會到這裡，將tag保存到scale_tags當中
                if isinstance(aggregated_tags, list):
                    scale_tags_list.extend(aggregated_tags)
                else:
                    scale_tags_list.append(aggregated_tags)

            # 將熱力圖進行保存
            if isinstance(aggregated_heatmaps, list):
                scale_heatmaps_list.extend(aggregated_heatmaps)
            else:
                scale_heatmaps_list.append(aggregated_heatmaps)

        # 將不同尺度的熱力圖進行融合，aggregated_heatmaps shape = [batch_size, num_joints, height, width]
        aggregated_heatmaps = aggregate_scale(
            scale_heatmaps_list,
            align_corners=self.test_cfg.get('align_corners', True),
            aggregate_scale='average')

        # 將不同tags進行融合，aggregated_tags shape = [batch_size, num_joints, height, width, scale * 2]
        aggregated_tags = aggregate_scale(
            scale_tags_list,
            align_corners=self.test_cfg.get('align_corners', True),
            aggregate_scale='unsqueeze_concat')

        # 獲取熱力圖的高寬資料
        heatmap_size = aggregated_heatmaps.shape[2:4]
        # 獲取tag的高寬資料
        tag_size = aggregated_tags.shape[2:4]
        if heatmap_size != tag_size:
            # 如果熱力圖的高寬與tag的高寬不同就會到這裡進行高寬調整
            # 構建一個臨時list
            tmp = []
            # 遍歷tags資料，這裡tags的資料數量會是保存在shape[-1]的地方
            for idx in range(aggregated_tags.shape[-1]):
                tmp.append(
                    # 透過差值方式進行調整，會將大小調整到與熱力圖相同
                    torch.nn.functional.interpolate(
                        aggregated_tags[..., idx],
                        size=heatmap_size,
                        mode='bilinear',
                        align_corners=self.test_cfg.get('align_corners', True)).unsqueeze(-1))
            # 最後用concat進行拼接，會在最後一個維度進行
            aggregated_tags = torch.cat(tmp, dim=-1)

        # perform grouping
        # 進行資料解析
        # grouped = list[ndarray]，list長度會batch_size，ndarray shape [people, num_joints, (x, y, score, tag[0], tag[1])]
        # scores = list[float32]，list長度會batch_size，值會是整體平均置信度
        grouped, scores = self.parser.parse(aggregated_heatmaps,
                                            aggregated_tags,
                                            self.test_cfg['adjust'],
                                            self.test_cfg['refine'])

        # 透過get_group_preds獲取preds
        preds = get_group_preds(
            grouped,
            center,
            scale, [aggregated_heatmaps.size(3),
                    aggregated_heatmaps.size(2)],
            use_udp=self.use_udp)

        # 構建image_paths資料
        image_paths = list()
        # 將原圖像的檔案位置保存進去
        image_paths.append(img_metas['image_file'])

        if return_heatmap:
            # 如果有需要回傳熱力圖就會將預測的熱力圖轉成ndarray進行保存
            output_heatmap = aggregated_heatmaps.detach().cpu().numpy()
        else:
            # 否則就會是None
            output_heatmap = None

        # 保存資料到result當中
        result['preds'] = preds
        result['scores'] = scores
        result['image_paths'] = image_paths
        result['output_heatmap'] = output_heatmap

        return result

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='AssociativeEmbedding')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color=None,
                    pose_kpt_color=None,
                    pose_link_color=None,
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized image only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        pose_result = []
        for res in result:
            pose_result.append(res['keypoints'])

        imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                         pose_kpt_color, pose_link_color, radius, thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
