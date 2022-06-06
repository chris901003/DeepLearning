# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR(nn.Module):
    # 由下面進行實例化
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        # 可以看一下這裡，說明的很清楚了
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        # 已看過
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        # hidden_dim = 每個特徵圖上的點用多少維度的向量表示
        hidden_dim = transformer.d_model
        # 估計是拿來預測類別的
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # 透過多層感知機獲得
        # Mlp(input_dim, hidden_dim, output_dim, num_layers)
        # 就是FFN(hidden_dim = transformer每個的channel, 4 = output_dim, 3 = num_layers)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # 創建embedding大小為num_queries每個queries由hidden_dim維度的向量表示
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # 在把CNN出來的展平特徵圖輸入到transformer前需要先把channel變成hidden_dim的大小
        # 用kernel_size=1的Conv做降維
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # 已看過
        # samples = NestedTensor格式裡面有已經打包好的圖像tensor以及mask
        # samples的內部細節可以看上面的英文
        if isinstance(samples, (list, torch.Tensor)):
            # 正常來說不會到這裡來，因為在collect_fn就已經處理好了
            samples = nested_tensor_from_tensor_list(samples)
        # ----------------------------------------------------------------------------
        # 由CNN backbone提取特徵
        # features (List[NestedTensor])
        # pos (List[tensor]) tensor shape [batch_size, batch_size, channel, w, h]
        # ----------------------------------------------------------------------------
        features, pos = self.backbone(samples)

        # features的最後一個，也就是backbone最後的輸出，把tensor與mask拆出來
        src, mask = features[-1].decompose()
        assert mask is not None
        # ----------------------------------------------------------------------------
        # input_proj = backbone最後一層輸出的時後如果是resnet50的話channel會是2048
        # 這裡我們用1*1的卷積降維成256，讓後續的transformer可以用
        # mask = 就是mask，還是一樣True的地方表示填充，False表示非填充
        # query_embed = decoder的query的embedding，加上.weight後把embed的值拿出來(torch.nn.parameter.Parameter)
        # pos[-1] = 把第一個batch_size維度拿掉，不然會有兩層的batch_size
        # ----------------------------------------------------------------------------

        # ----------------------------------------------------------------------------
        # hs = [layers, batch_size, num_queries, channel] => decoder輸出
        # memory = [batch_size, channel, w * h] -> [batch_size, channel, w, h] => encoder輸出
        # 這裡我們只拿hs的結果，也就是只需要decoder的輸出結果
        # ----------------------------------------------------------------------------
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        # 進行類別預測的fc，將原先channel降維成num_classes+1
        # outputs_class shape [layers, batch_size, num_queries, num_classes + 1]
        outputs_class = self.class_embed(hs)
        # outputs_coord shape [layers, batch_size, num_queries, 4]
        # 這裡輸出還會再通過sigmoid，讓值都在0到1之間，對應到原圖上的相對座標
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # 將輔助輸出以及最後輸出分開，這裡out用dict的格式
        # outputs_class, outputs_coord的最後一層會是最後輸出
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # 如果有啟用輔助訓練的話就會進去
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        # 最後回傳
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 已看過
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # 就是除了最後一個輸出，其他輸出都是輔助輸出
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    # 由下面部分實例化
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        # 已看過
        """ Create the criterion.
        # 可以看一下這裡的變數解釋
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        # losses = ['labels', 'boxes', 'cardinality']
        self.losses = losses
        # empty_weight暫時不知道要做什麼用
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # 已看過
        # indices
        # List[tuple(row, col)]，List長度就是batch_size大小
        # [num_queries, gt_box_for_index_i]
        # row指的就是num_queries中的哪個，col就是說這張照片中的那個gt_box
        assert 'pred_logits' in outputs
        # 拿出outputs中的最後分類類別輸出
        src_logits = outputs['pred_logits']

        # idx = (batch_idx, src_idx)
        # batch_idx = 把一個batch的gt_box concat起來所以長度是所有的gt_box，只是batch中第一張圖片的會是0，第二張會是1，後面以此類推
        # src_idx = 把一個batch有對應到gt_box的query concat起來
        idx = self._get_src_permutation_idx(indices)
        # 我們遍歷整個targets以及indices，找到targets中的labels，透過indices在col的紀錄我們可以知道要拿哪個index的label
        # 最後把整個batch的結果concat起來
        # target_classes_o shape [total_number_match_gt_box]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # 構建一個shape [batch_size, num_queries]且填充值為self.num_classes的tensor
        # self.num_classes = 當作背景，因為我們是要預測的值加1，這個多出來的就是背景
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # 用batch_idx找到是哪個batch再用src_idx找到對應的query，讓正確答案變成是target_classes_o的值
        # 這樣就生成了正確答案的表
        target_classes[idx] = target_classes_o

        # 根據pytorch官方，cross_entropy需要放入的shape [batch_size, classes, ...]，所以我們這裡需要transpose一下
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        # 保存下來
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # src_logits[idx] = 有對應上gt_box的query對於每個分類類別的預測值
            # target_classes_o = 正確分類類別
            # src_logits shape [total_number_match_gt_box, num_classes]
            # target_classes_o shape [total_number_match_gt_box]
            # accuracy回傳的是一個list但因為我們只在意top1所以只會取[0]這個資料
            # accuracy的值表示正確率而且是以100%的格式，所以減100就是錯誤率
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        # 最後回傳的loss當中會有兩個key
        # loss_ce = 用交叉商計算出來的損失值
        # class_error = 錯誤率，以100%格式
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        # 已看過
        # 拿出分類類別預測shape [total_number_match_gt_box, num_classes]
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        # 取出每張圖片的gt_box數量shape [batch_size]
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        # card_pred = 計算預測時有多少預測為正樣本，也就是預測為背景的概率不是最大
        # card_pred shape [batch_size]
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # 使用預測為正樣本的數量與真實正樣本的數量計算l1損失
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # 已看過
        # indices
        # List[tuple(row, col)]，List長度就是batch_size大小
        # [num_queries, gt_box_for_index_i]
        # row指的就是num_queries中的哪個，col就是說這張照片中的那個gt_box
        assert 'pred_boxes' in outputs
        # idx = (batch_idx, src_idx)
        # batch_idx = 把一個batch的gt_box concat起來所以長度是所有的gt_box，只是batch中第一張圖片的會是0，第二張會是1，後面以此類推
        # src_idx = 把一個batch有對應到gt_box的query concat起來
        idx = self._get_src_permutation_idx(indices)
        # 把有對應到gt_box的query拿出來
        # src_boxes shape [total_number_match_gt_box, 4]
        src_boxes = outputs['pred_boxes'][idx]
        # 取出正確人工標記的gt_box，這裡index是跟上面的src_boxes有對應上的，也就是相同index是互相對應上的
        # target_boxes shape [total_number_match_gt_box, 4]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 用src_boxes以及target_boxes計算l1損失
        # loss_bbox shape [total_number_match_gt_box, 4]
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        # 對loss_bbox做總和再除以數量取得平均值
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # box_ops.generalized_box_iou = 需要將輸入的格式調整為(xmin, ymin, xmax, ymax)然後進行giou計算
        # diag = 我們只對對角線上的值有需求，因為這才是query對應上gt_box的giou其他的都不是我們要的
        # giou loss = 1 - giou
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        # 計算總和再取平均值
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        # 最後回傳的loss有兩個key
        # loss_bbox = l1損失
        # loss_giou = giou損失
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # 未讀
        # 如果要訓練segmentation才會用到
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # 已看過
        # indices
        # List[tuple(row, col)]，List長度就是batch_size大小
        # [num_queries, gt_box_for_index_i]
        # row指的就是num_queries中的哪個，col就是說這張照片中的那個gt_box
        # permute predictions following indices
        # batch_idx = 對於每張圖做出一個長度一樣且數字為在batch中第幾張圖的數字
        # Ex: batch_size = 3，第一張有3個gt_box，第二張有2個gt_box，第三張有4個gt_box
        # 那麼batch_idx = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # 把每張圖片有對應上gt_box的queries拼接在一起
        src_idx = torch.cat([src for (src, _) in indices])
        # return batch_idx, src_idx shape [total_number_match_gt_box]
        # 其實total_number_match_gt_box也會等於這個batch中的gt_box總合，因為queries總共有100個，圖中的gt_box一定可以透過
        # 匈牙利算法找到總損失值最小的query
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # 未讀
        # 如果要訓練segmentation才會用到
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # 已看過
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        # 判斷要調用哪個loss計算方法
        # 基本上kwargs裡面不會有東西
        # masks是在訓練segmentation才會用到這裡我們不會用到
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs (Dict)
        # {
        #   'pred_logits': shape [batch_size, num_queries, num_classes + 1],
        #   'pred_boxes': shape [batch_size, num_queries, 4],
        #   'aux_outputs': List[{
        #                           'pred_logits': shape [batch_size, num_queries, num_classes + 1],
        #                           'pred_boxes': shape [batch_size, num_queries, 4]
        #                      }]
        # }
        # targets = [batch_size, annotations_per_image] = [tuple[list[dict]]] = [幾張照片[每張照片的標籤信息[每個標籤的內容]]]
        # 已看過

        # 將最後輸出拿出來，也就是不要輔助輸出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # 匈牙利匹配，去看看matcher中HungarianMatcher的forward
        # List[tuple(row, col)]，List長度就是batch_size大小
        # [num_queries, gt_box_for_index_i]
        # row指的就是num_queries中的哪個，col就是說這張照片中的那個gt_box
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # 這次batch_size中所有圖片的gt_box數量
        num_boxes = sum(len(t["labels"]) for t in targets)
        # 轉成tensor
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            # 多gpu會用到的
            # Reduces the tensor data across all machines in such a way that all get the final result.
            torch.distributed.all_reduce(num_boxes)
        # 均攤一下，把所有匡的數量除以gpu數量
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        # 計算損失
        # self.losses = ['labels', 'boxes', 'cardinality']
        losses = {}
        # ---------------------------------------------------------
        # 'labels':
        # loss_ce = 用交叉商計算出來的損失值
        # class_error = 錯誤率，以100%格式
        # 'boxes':
        # loss_bbox = l1損失
        # loss_giou = giou損失
        # 'cardinality':
        # cardinality_error = 正負樣本匹配損失
        # ---------------------------------------------------------
        for loss in self.losses:
            # 根據要計算的loss放入get_loss中
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # 計算輔助訓練的loss
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # 使用匈牙利算法獲得哪個query應該要對應上哪個gt_box
                # 有問題可以往上看
                indices = self.matcher(aux_outputs, targets)
                # 一樣去計算我們各種的損失，self.losses = ['labels', 'boxes', 'cardinality']
                for loss in self.losses:
                    # 如果是訓練segmentation會有masks loss，這裡因為計算masks loss太花費時間所以直接跳過
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # 計算labels時會順便計算class_error表示錯誤率，那部分是用來print用的，而中間層輸出的我們就不用print所以取消
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    # 剩下的部分就跟上面在計算最後輸出的損失是一樣的，只是在key的地方會加上_{i}，來分辨是哪個層的loss
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 這個是return losses dict裡面會有的東西，如果有加上輔助損失的就會多一些一樣的key只是後面會多_{i}，來分辨是哪個層的loss
        # ---------------------------------------------------------
        # {
        #   loss_ce:用交叉商計算出來的損失值,
        #   class_error:錯誤率，以100%格式,
        #   loss_bbox:l1損失,
        #   loss_giou:giou損失,
        #   cardinality_error:正負樣本匹配損失,
        #   ...
        # }
        # ---------------------------------------------------------
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # ---------------------------------------------------------
        # outputs (Dict)
        # {
        #   'pred_logits': shape [batch_size, num_queries, num_classes + 1],
        #   'pred_boxes': shape [batch_size, num_queries, 4]
        # ---------------------------------------------------------
        # 預測匡與預測類別拿出來
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # 檢查batch_size要是一樣的以及target_size是高和寬
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # 對預測類別輸出做softmax
        prob = F.softmax(out_logits, -1)
        # 這裡是把背景排除掉，但是在做softmax時背景也有參與
        # scores = 一個query中最高分數的分數
        # labels = 一個query中最高分數的index
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        # 格式轉換 [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        # 預測出來的是相對座標這裡我們轉換成絕對座標
        # 在第一個維度上面做拆解，img_h和img_w shape [batch_size]
        img_h, img_w = target_sizes.unbind(1)
        # scale_fct shape [batch_size, 4]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # boxes shape [batch_size, num_queries, 4]，所以這裡我們scale_fct需要在num_queries上面做擴維
        boxes = boxes * scale_fct[:, None, :]

        # ---------------------------------------------------------
        # 將結果存成字典
        # scores shape [batch_size, num_queries]
        # labels shape [batch_size, num_queries]
        # boxes shape [batch_size, num_queries, 4]
        # ---------------------------------------------------------
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        # 已看過
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # 就是FFN
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        # 已看過
        # x shape [layers, batch_size, num_queries, channel]
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        # return x shape [layers, batch_size, num_queries, output_dim]
        return x


def build(args):
    # 已看過
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # ----------------------------------------------------------------------------
    # num_classes = 要預測的目標數量再加一，以VOC為例我們就需要把num_classes設定成21
    # coco_panoptic是全景分析的數據集
    # ----------------------------------------------------------------------------
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250

    device = torch.device(args.device)

    # ----------------------------------------------------------------------------
    # 構建backbone
    # ----------------------------------------------------------------------------
    backbone = build_backbone(args)
    # ----------------------------------------------------------------------------
    # 構建transformer
    # ----------------------------------------------------------------------------
    transformer = build_transformer(args)

    # ----------------------------------------------------------------------------
    # 構建DETR
    # backbone = CNN
    # transformer = 拿到特徵圖後的處理
    # num_classes = 分類數
    # num_queries = 一張圖片會有多少個匡
    # aux_loss = decoder中每層的輸出要不要算入loss
    # ----------------------------------------------------------------------------
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    # 預設為False除非要訓練segmentation
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    # 實例化匈牙利匹配的東西
    matcher = build_matcher(args)
    # loss coefficient
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    # 如果是要訓練segmentation會有多下面兩個loss coefficient
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    # 要用每層decoder輸出進行loss計算
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            # 每一層都會有一份跟上面weight_dict一樣的東西，這裡會複製decoder_layers-1次
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # 紀錄有哪幾種的loss
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # 設定標準
    # matcher = 上面實例化的匈牙利匹配
    # eos_coef = classification weight of the no-object class
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    # PostProcess就只有forward函數
    postprocessors = {'bbox': PostProcess()}
    # 訓練Segmentation才會用到
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    # model = 我們需要用到的預測模型
    # criterion = 預測後處理
    # postprocessors = 將輸出變成coco api想要的格式
    return model, criterion, postprocessors
