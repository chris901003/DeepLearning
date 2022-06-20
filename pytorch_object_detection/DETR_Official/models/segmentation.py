# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
import io
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

import util.box_ops as box_ops
from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


class DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        """
        :param detr: 生成好的detr模型
        :param freeze_detr: 如果有加載預訓練權重就會是True，否則就是False
        """
        # 已看過
        super().__init__()
        self.detr = detr

        # 如果有加載權重就可以凍結
        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        # 獲取transformer的channel深度以及多頭注意力機制中的頭數
        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        # 實例化了兩個東西
        # This is a 2D attention module, which only returns the attention softmax (no multiplication by value)
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

    def forward(self, samples: NestedTensor):
        # 已看過
        # 如果對NestedTensor有問題可以到misc.py裡面看
        if isinstance(samples, (list, torch.Tensor)):
            # 如果不是NestedTensor格式就進行轉換，但是理論上不會不是NestedTensor格式
            samples = nested_tensor_from_tensor_list(samples)
        # samples = {
        #   'mask': shape [batch_size, height, width],
        #   'tensors': shape [batch_size, 3, height, width]
        # }

        # 先放入detr中的backbone做特徵提取
        # ----------------------------------------------------------------------------
        # features = (List[NestedTensor])，list長度就是從backbone拿出多少層的輸出
        # pos = (List[tensor]) tensor shape [batch_size, channel, height, width]，list長度就是從backbone拿出多少層的輸出
        # ----------------------------------------------------------------------------
        features, pos = self.detr.backbone(samples)

        # 獲取batch_size
        bs = features[-1].tensors.shape[0]

        # 將最後一層的輸出拆解出來，將特徵圖與mask分開
        src, mask = features[-1].decompose()
        assert mask is not None
        # 透過卷積調整channel到transformer需要的channel深度
        # src_proj shape [batch_size, hidden_dim, height, width]
        src_proj = self.detr.input_proj(src)
        # 輸入到transformer當中
        # pos[-1]表示拿出最後輸出層的位置編碼
        # ----------------------------------------------------------------------------
        # hs = [layers, batch_size, num_queries, channel] => decoder輸出，layers表示decoder的多層輸出
        # memory = [batch_size, channel, w, h] => encoder輸出
        # ----------------------------------------------------------------------------
        hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])

        # outputs_class shape [layers, batch_size, num_queries, num_classes + 1]
        outputs_class = self.detr.class_embed(hs)
        # outputs_coord shape [layers, batch_size, num_queries, 4]
        # 透過sigmoid將數值控制在[0, 1]，所以這裡是相對座標
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        # 取出decoder最後一層輸出當作主輸出
        # outputs_class shape [batch_size, num_queries, num_classes + 1]
        # output_coord shape [batch_size, num_queries, 4]
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        # 之後再用輔助訓練的輸出
        if self.detr.aux_loss:
            out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)

        # FIXME h_boxes takes the last one computed, keep this in mind
        # bbox_mask shape [batch_size, num_queries, num_heads, height, width]
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)

        # 將encoder調整完channel的輸出以及上方的輸出以及backbone的中間層特徵圖輸出輸入進去
        # seg_masks shape [batch_size * num_queries, 1, height, width]
        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        # outputs_seg_masks shape [batch_size, num_queries, height, width]
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])

        # 將結果存入到out當中且key名稱pred_masks
        out["pred_masks"] = outputs_seg_masks
        return out


def _expand(tensor, length: int):
    # 已看過
    # 先第一維度擴維，之後再擴維地方決定數字，最後將第0以及1做展平
    # length = num_queries
    # tensor shape [batch_size, hidden_dim, height, width]
    # [batch_size, hidden_dim, height, width] -> [batch_size, 1, hidden_dim, height, width] ->
    # [batch_size, num_queries, hidden_dim, height, width] -> [batch_size * num_queries, hidden_dim, height, width]
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        """
        :param dim: hidden_dim + nheads
        :param fpn_dims: [1024, 512, 256]，這個是backbone中間層輸出的channel深度
        :param context_dim: hidden_dim
        """
        # hidden_dim預設為256，nheads預設為8
        # 已看過
        super().__init__()

        # inter_dims = [264, 128, 64, 32, 16, 4]
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        # 一直往下就會讓channel維度變成1
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        # GroupNorm = 一種channel標準化方式，傳入的參數(num_groups, num_channels)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        # 其他的一些卷積
        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        # 初始化權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        """
        :param x: [batch_size, hidden_dim, height, width]
        :param bbox_mask: [batch_size, num_queries, num_heads, height, width]
        :param fpns: [[batch_size, channel, height, width], ..., []]，長度就是backbone中間層的數量
        :return: shape [batch_size * num_queries, 1, height, width]
        """
        # 已看過
        # _expand return shape [batch_size * num_queries, hidden_dim, height, width]
        # bbox_mask flatten shape [batch_size * num_queries, num_heads, height, width]
        # x shape [batch_size * num_queries, hidden_dim + num_heads, height, width]
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        # x shape [batch_size * num_queries, hidden_dim + num_heads, height, width]
        x = self.lay1(x)
        # group標準化
        x = self.gn1(x)
        # 通過激活函數
        x = F.relu(x)
        # x shape [batch_size * num_queries, 128, height, width]
        x = self.lay2(x)
        # group標準化
        x = self.gn2(x)
        # 通過激活函數
        x = F.relu(x)

        # 在這裡的fpns高寬大小是從小到大，所以都是對x進行雙線性差值
        # fpns[0] shape [batch_size, 1024, height, width]
        # cur_fpn shape [batch_size, 128, height, width]
        cur_fpn = self.adapter1(fpns[0])
        # 正常來說cur_fpn.size(0)與x.size(0)會不相同，所以會進去做expand
        if cur_fpn.size(0) != x.size(0):
            # cur_fpn shape [batch_size * num_queries, 128, height, width]
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 在做相加之前先對x進行雙線性插值，讓高寬變成與cur_fpn相同這樣才可以相加
        # x shape [batch_size * num_queries, 128, height, width]
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # x shape [batch_size * num_queries, 64, height, width]
        x = self.lay3(x)
        # group標準化
        x = self.gn3(x)
        # 通過激活函數
        x = F.relu(x)

        # fpns[1] shape [batch_size, 512, height, width]
        # cur_fpn shape [batch_size, 64, height, width]
        cur_fpn = self.adapter2(fpns[1])
        # 同樣是擴增維度後再做展平
        if cur_fpn.size(0) != x.size(0):
            # cur_fpn shape [batch_size * num_queries, 64, height, width]
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 對x做雙線性插值，讓高寬進行擴增，最後再做相加
        # x shape [batch_size * num_queries, 64, height, width]
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # x shape [batch_size * num_queries, 32, height, width]
        x = self.lay4(x)
        # group標準化
        x = self.gn4(x)
        # 通過激活函數
        x = F.relu(x)

        # fpns[2] shape [batch_size, 256, height, width]
        # cur_fpn shape [batch_size, 32, height, width]
        cur_fpn = self.adapter3(fpns[2])
        # 與上面都相同了
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # x shape [batch_size * num_queries, height, width]
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # x shape [batch_size * num_queries, 16, height, width]
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        # x shape [batch_size * num_queries, 1, height, width]
        x = self.out_lay(x)
        # return shape [batch_size * num_queries, 1, height, width]
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        """
        :param query_dim: 這裡傳入的是transformer中的channel深度
        :param hidden_dim: 與query_dim相同
        :param num_heads: 多頭注意力機制的頭數
        :param dropout: dropout rate
        :param bias: 是否啟用偏至
        """
        # 已看過
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # query與key產生
        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        # 將bias先都設置為0
        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        # 初始化權重
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        # 多頭注意力機制中q*k後需要乘上的值
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        """
        :param q: [batch_size, num_queries, channel] => decoder最後一層輸出
        :param k: [batch_size, channel, height, width] => encoder輸出
        :param mask: [batch_size, height, width]
        :return:
        """
        # 已看過
        # q shape [batch_size, num_queries, hidden_dim]，hiddden_dim預設為256，也就是transformer中的channel深度
        q = self.q_linear(q)
        # 根據pytorch官方F.conv2d需要帶入的參數為(input, weight, bias, stride, padding)
        # weight = [out_channels, in_channel / groups, kH, kW]
        # k_linear.weight shape [query_dim, hidden_dim]，在這裡query_dim與hidden_dim相同
        # weight shape [query_dim, hidden_dim, 1, 1]
        # k shape [batch_size, hidden_dim, height, width]
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        # qh shape [batch_size, num_queries, num_heads, hidden_dim / num_heads]
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        # kh shape [batch_size, num_heads, hidden_dim / num_heads, height, width]
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        # qh * self.normalize_fact標準化用的
        # einsum = 愛因斯坦求和約定，詳細上網查一下，這個有點難說明
        # weights shape [batch_size, num_queries, num_heads, height, width]
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        # 在mask值為1的地方會是-inf，值為0的地方會是weights原先的值
        # 我們知道mask中為1的地方表示是padding出來的，所以合理
        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        # 先在第二維度上展平後對展平部分做softmax之後再將shape調整回原來的樣子，所以最後shape不會有改變
        # 因為上面有將padding部分設定成-inf，所以經過softmax後這些padding部分會無限接近0
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        # return shape [batch_size, num_queries, num_heads, height, width]
        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # 已看過
    # 用來可視化用的
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    # 已看過
    # inputs shape [total_number_match_gt_box, height * width]
    # targets shape [total_number_match_gt_box, height * width]
    prob = inputs.sigmoid()
    # 進行二值交叉熵損失計算，裡面會先進行softmax再計算
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 再對計算出來的ce_loss進行調整
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # 最後輸出loss
    return loss.mean(1).sum() / num_boxes


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        # 已看過
        # 在detr中被實例化
        super().__init__()
        # 記錄下閾值
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        # ---------------------------------------------------------
        # results = {
        #   'scores':[batch_size, num_queries],
        #   'labels':[batch_size, num_queries],
        #   'boxes':[batch_size, num_queries,4]
        # }
        # ---------------------------------------------------------
        # outputs (Dict)
        # {
        #   'pred_logits': shape [batch_size, num_queries, num_classes + 1],
        #   'pred_boxes': shape [batch_size, num_queries, 4],
        #   'pred_masks': shape[batch_size, num_queries, height, width](只有在訓練segmentation中才會有)
        # }
        # ---------------------------------------------------------
        # orig_target_sizes, max_target_sizes shape [batch_size, 2]
        assert len(orig_target_sizes) == len(max_target_sizes)
        # 一個batch當中高寬最大的大小
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        # outputs_masks shape [batch_size, num_queries, height, width]
        outputs_masks = outputs["pred_masks"].squeeze(2)
        # 使用雙線性差值，將預測圖縮放回(max_h,max_w)大小
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        # 透過sigmoid只要超過閾值表示那個地方是有目標的，我們會用True表示，也就是之後會在True的地方進行圖色
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        # 遍歷整個batch，在原先的results中添加masks內容
        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            # results[i]['masks'] shape [num_queries, 1, img_h, img_w]，擴維的部分就是channel，因為之後再調色所以這裡是單通道
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            # 再透過雙線性差值將大小調整到與原圖大小相同
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()

        # 最後return results masks shape [num_queries, 1, orig_height, orig_width]
        return results


class PostProcessPanoptic(nn.Module):
    """This class converts the output of the model to the final panoptic result, in the format expected by the
    coco panoptic API """

    def __init__(self, is_thing_map, threshold=0.85):
        """
        Parameters:
           is_thing_map: This is a whose keys are the class ids, and the values a boolean indicating whether
                          the class is  a thing (True) or a stuff (False) class
           threshold: confidence threshold: segments with confidence lower than this will be deleted
        """
        super().__init__()
        self.threshold = threshold
        self.is_thing_map = is_thing_map

    def forward(self, outputs, processed_sizes, target_sizes=None):
        """ This function computes the panoptic prediction from the model's predictions.
        Parameters:
            outputs: This is a dict coming directly from the model. See the model doc for the content.
            processed_sizes: This is a list of tuples (or torch tensors) of sizes of the images that were passed to the
                             model, ie the size after data augmentation but before batching.
            target_sizes: This is a list of tuples (or torch tensors) corresponding to the requested final size
                          of each prediction. If left to None, it will default to the processed_sizes
            """
        if target_sizes is None:
            target_sizes = processed_sizes
        assert len(processed_sizes) == len(target_sizes)
        out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
        assert len(out_logits) == len(raw_masks) == len(target_sizes)
        preds = []

        def to_tuple(tup):
            if isinstance(tup, tuple):
                return tup
            return tuple(tup.cpu().tolist())

        for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
            out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
        ):
            # we filter empty queries and detection below threshold
            scores, labels = cur_logits.softmax(-1).max(-1)
            keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self.threshold)
            cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
            cur_scores = cur_scores[keep]
            cur_classes = cur_classes[keep]
            cur_masks = cur_masks[keep]
            cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
            cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

            h, w = cur_masks.shape[-2:]
            assert len(cur_boxes) == len(cur_classes)

            # It may be that we have several predicted masks for the same stuff class.
            # In the following, we track the list of masks ids for each stuff class (they are merged later on)
            cur_masks = cur_masks.flatten(1)
            stuff_equiv_classes = defaultdict(lambda: [])
            for k, label in enumerate(cur_classes):
                if not self.is_thing_map[label.item()]:
                    stuff_equiv_classes[label.item()].append(k)

            def get_ids_area(masks, scores, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = masks.transpose(0, 1).softmax(-1)

                if m_id.shape[-1] == 0:
                    # We didn't detect any mask :(
                    m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
                else:
                    m_id = m_id.argmax(-1).view(h, w)

                if dedup:
                    # Merge the masks corresponding to the same stuff class
                    for equiv in stuff_equiv_classes.values():
                        if len(equiv) > 1:
                            for eq_id in equiv:
                                m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

                final_h, final_w = to_tuple(target_size)

                seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
                seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

                np_seg_img = (
                    torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
                )
                m_id = torch.from_numpy(rgb2id(np_seg_img))

                area = []
                for i in range(len(scores)):
                    area.append(m_id.eq(i).sum().item())
                return area, seg_img

            area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
            if cur_classes.numel() > 0:
                # We know filter empty masks as long as we find some
                while True:
                    filtered_small = torch.as_tensor(
                        [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                    )
                    if filtered_small.any().item():
                        cur_scores = cur_scores[~filtered_small]
                        cur_classes = cur_classes[~filtered_small]
                        cur_masks = cur_masks[~filtered_small]
                        area, seg_img = get_ids_area(cur_masks, cur_scores)
                    else:
                        break

            else:
                cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

            segments_info = []
            for i, a in enumerate(area):
                cat = cur_classes[i].item()
                segments_info.append({"id": i, "isthing": self.is_thing_map[cat], "category_id": cat, "area": a})
            del cur_classes

            with io.BytesIO() as out:
                seg_img.save(out, format="PNG")
                predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
            preds.append(predictions)
        return preds
