# ------------------------------------------------------------------------------
# Adapted from https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from ..builder import LOSSES


def _make_input(t, requires_grad=False, device=torch.device('cpu')):
    """Make zero inputs for AE loss.

    Args:
        t (torch.Tensor): input
        requires_grad (bool): Option to use requires_grad.
        device: torch device

    Returns:
        torch.Tensor: zero input.
    """
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    inp = inp.to(device)
    return inp


@LOSSES.register_module()
class HeatmapLoss(nn.Module):
    """Accumulate the heatmap loss for each image in the batch.

    Args:
        supervise_empty (bool): Whether to supervise empty channels.
    """

    def __init__(self, supervise_empty=True):
        """ 構建熱力圖損失計算
        Args:
            supervise_empty: 是否需要監測空的channel
        """
        # 繼承自nn.Module，對繼承對象劑型初始化
        super().__init__()
        # 將supervise_empty保存
        self.supervise_empty = supervise_empty

    def forward(self, pred, gt, mask):
        """Forward function.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred (torch.Tensor[N,K,H,W]):heatmap of output.
            gt (torch.Tensor[N,K,H,W]): target heatmap.
            mask (torch.Tensor[N,H,W]): mask of target.
        """
        """ 計算熱力圖損失的forward函數
        Args:
            pred: 預測出來的結果，tensor shape [batch_size, num_joints, height, width]
            gt: 標註的熱力圖，tensor shape [batch_size, num_joints, height, width]
            mask: 紀錄哪些地方不用計算損失值，tensor shape [batch_size, height, width]
        """
        # gt = 熱力圖標註，每個關節點會有自己的一張高寬圖，如果原始圖像當中有多個人就會在該熱力圖上有多個點
        #      例如在左眼的熱力圖上就會有兩個高斯分佈的圖

        # 檢查pred與gt的size是否完全相同，如果不相同就會報錯
        assert pred.size() == gt.size(), f'pred.size() is {pred.size()}, gt.size() is {gt.size()}'

        if not self.supervise_empty:
            # 如果沒有需要supervise_empty就會到這裡
            empty_mask = (gt.sum(dim=[2, 3], keepdim=True) > 0).float()
            loss = ((pred - gt)**2) * empty_mask.expand_as(
                pred) * mask[:, None, :, :].expand_as(pred)
        else:
            # 如果有需要進行supervise_empty就會到這裡
            # 計算損失值，透過mask會將不需要計算的部分用0做相乘
            loss = ((pred - gt)**2) * mask[:, None, :, :].expand_as(pred)
        # 最後全部求平均值
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        # 回傳最後的loss值
        return loss


@LOSSES.register_module()
class AELoss(nn.Module):
    """Associative Embedding loss.

    `Associative Embedding: End-to-End Learning for Joint Detection and
    Grouping <https://arxiv.org/abs/1611.05424v2>`_.
    """

    def __init__(self, loss_type):
        """ 構建關聯嵌入損失
        Args:
            loss_type: 計算方式
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        # 保存loss_type
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):
        """Associative embedding loss for one image.

        Note:
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred_tag (torch.Tensor[KxHxW,1]): tag of output for one image.
            joints (torch.Tensor[M,K,2]): joints information for one image.
        """
        """ 計算一張圖像的關聯嵌入損失
        Args:
            pred_tag: 預測的結果，tensor shape [num_joints * height * width, 1]
            joints: 標註關節點座標，tensor shape [max_people, num_joints, 2]
        """

        # 保存tags的list
        tags = []
        # 將pull設定成0
        pull = 0
        # 遍歷圖像當中的人
        for joints_per_person in joints:
            # 將tmp清空
            tmp = []
            # 遍歷每個人的關節點
            for joint in joints_per_person:
                if joint[1] > 0:
                    # 如果joints[1]大於0表示該點可以被檢測出來，將該點的座標值放到tmp當中
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                # 如果tmp的長度是0就直接跳過，表示沒有任何關節點可以檢測到
                continue
            # 將tmp資料堆疊起來，tensor shape [合法關節點, 1]
            tmp = torch.stack(tmp)
            # 將取出的預測值取平均後保存到tags當中
            tags.append(torch.mean(tmp, dim=0))
            # 將tmp當中的值減去平均值平方後取平均加到pull當中
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

        # num_tags = 圖中有多少人
        num_tags = len(tags)
        # 計算之後的損失
        if num_tags == 0:
            return (
                _make_input(torch.zeros(1).float(), device=pred_tag.device),
                _make_input(torch.zeros(1).float(), device=pred_tag.device))
        elif num_tags == 1:
            return (_make_input(
                torch.zeros(1).float(), device=pred_tag.device), pull)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unknown ae loss type')

        push_loss = push / ((num_tags - 1) * num_tags) * 0.5
        pull_loss = pull / (num_tags)

        return push_loss, pull_loss

    def forward(self, tags, joints):
        """Accumulate the tag loss for each image in the batch.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            tags (torch.Tensor[N,KxHxW,1]): tag channels of output.
            joints (torch.Tensor[N,M,K,2]): joints information.
        """
        """ 計算標籤損失
        Args:
            tags: 預測的結果，tensor shape [batch_size, num_joints * height * width, 1]
            joints: 標註的關節點資料，tensor shape [batch_size, max_people, num_joints, 2]
        """

        # 構建pushes與pulls的損失list
        pushes, pulls = [], []
        # 將joints資料轉到cpu當中並且轉成ndarray格式
        joints = joints.cpu().data.numpy()
        # 獲取當前batch大小
        batch_size = tags.size(0)
        # 遍歷batch的大小
        for i in range(batch_size):
            # 透過singleTagLoss計算push與pull的損失
            push, pull = self.singleTagLoss(tags[i], joints[i])
            # 將損失值保存起來
            pushes.append(push)
            pulls.append(pull)
        # 最後stack起來回傳
        return torch.stack(pushes), torch.stack(pulls)


@LOSSES.register_module()
class MultiLossFactory(nn.Module):
    """Loss for bottom-up models.

    Args:
        num_joints (int): Number of keypoints.
        num_stages (int): Number of stages.
        ae_loss_type (str): Type of ae loss.
        with_ae_loss (list[bool]): Use ae loss or not in multi-heatmap.
        push_loss_factor (list[float]):
            Parameter of push loss in multi-heatmap.
        pull_loss_factor (list[float]):
            Parameter of pull loss in multi-heatmap.
        with_heatmap_loss (list[bool]):
            Use heatmap loss or not in multi-heatmap.
        heatmaps_loss_factor (list[float]):
            Parameter of heatmap loss in multi-heatmap.
        supervise_empty (bool): Whether to supervise empty channels.
    """

    def __init__(self,
                 num_joints,
                 num_stages,
                 ae_loss_type,
                 with_ae_loss,
                 push_loss_factor,
                 pull_loss_factor,
                 with_heatmaps_loss,
                 heatmaps_loss_factor,
                 supervise_empty=True):
        """ 給bottom-up的模型使用的損失計算
        Args:
            num_joints: 關節點數量，也是分類數量
            num_stages: stage的數量
            ae_loss_type: ae損失的類別
            with_ae_loss: 哪些需要使用ae損失
            push_loss_factor: 熱力圖中推送損失的參數
            pull_loss_factor: 熱圖中拿取損失的參數
            with_heatmaps_loss: 是否使用熱力圖損失
            heatmaps_loss_factor: 熱力圖的損失參數
            supervise_empty: 是否監控空的channel
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()

        # 檢查with_heatmaps_loss是否為list或是tuple格式
        assert isinstance(with_heatmaps_loss, (list, tuple)), \
            'with_heatmaps_loss should be a list or tuple'
        # 檢查heatmaps_loss_factor是否為list或是tuple格式
        assert isinstance(heatmaps_loss_factor, (list, tuple)), \
            'heatmaps_loss_factor should be a list or tuple'
        # 檢查with_ae_loss是否為list或是tuple格式
        assert isinstance(with_ae_loss, (list, tuple)), \
            'with_ae_loss should be a list or tuple'
        # 檢查push_loss_factor是否為list或是tuple格式
        assert isinstance(push_loss_factor, (list, tuple)), \
            'push_loss_factor should be a list or tuple'
        # 檢查pull_loss_factor是否為list或是tuple格式
        assert isinstance(pull_loss_factor, (list, tuple)), \
            'pull_loss_factor should be a list or tuple'

        # 保存傳入的參數
        self.num_joints = num_joints
        self.num_stages = num_stages
        self.ae_loss_type = ae_loss_type
        self.with_ae_loss = with_ae_loss
        self.push_loss_factor = push_loss_factor
        self.pull_loss_factor = pull_loss_factor
        self.with_heatmaps_loss = with_heatmaps_loss
        self.heatmaps_loss_factor = heatmaps_loss_factor

        # 構建熱力圖損失函數
        self.heatmaps_loss = \
            nn.ModuleList(
                [
                    HeatmapLoss(supervise_empty)
                    if with_heatmaps_loss else None
                    for with_heatmaps_loss in self.with_heatmaps_loss
                ]
            )

        # 構建ae損失函數
        self.ae_loss = \
            nn.ModuleList(
                [
                    AELoss(self.ae_loss_type) if with_ae_loss else None
                    for with_ae_loss in self.with_ae_loss
                ]
            )

    def forward(self, outputs, heatmaps, masks, joints):
        """Forward function to calculate losses.

        Note:
            - batch_size: N
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K
            - output_channel: C C=2K if use ae loss else K

        Args:
            outputs (list(torch.Tensor[N,C,H,W])): outputs of stages.
            heatmaps (list(torch.Tensor[N,K,H,W])): target of heatmaps.
            masks (list(torch.Tensor[N,H,W])): masks of heatmaps.
            joints (list(torch.Tensor[N,M,K,2])): joints of ae loss.
        """
        """ 給bottom-up的模型使用的損失計算
        Args:
            outputs: 模型預測的輸出結果，list[tensor]，tensor shape [batch_size, channel, height, width]
            heatmaps: 標註熱力圖資料，list[tensor]，tensor shape [batch_size, num_joints, height, width]
            masks: 標註哪些地方不需紀錄損失值，list[tensor]，tensor shape [batch_size, height, width]
            joints: 關節點座標資訊，list[tensor]，tensor shape [batch_size, mas_people, num_joints, 2]
        """

        # 保存熱力圖的損失值
        heatmaps_losses = []
        # 保存push損失
        push_losses = []
        # 保存pull損失
        pull_losses = []
        # 遍歷預測輸出的數量，這裡如果有多尺度的輸出就會有多張預測圖
        for idx in range(len(outputs)):
            # 將offset_feat設定成0
            offset_feat = 0
            if self.heatmaps_loss[idx]:
                # 如果有實例化對應的heatmaps_loss對象就會到這裡，計算熱力圖的損失
                # 預測的熱力圖會是outputs當中在channel維度的前num_joints作為熱力圖的預測
                # heatmaps_pred shape = [batch_size, num_joints, height, width]
                heatmaps_pred = outputs[idx][:, :self.num_joints]
                # 將offset_feat設定成關節點數量
                offset_feat = self.num_joints
                # 將資料放入到熱力圖損失計算實例對象當中計算
                heatmaps_loss = self.heatmaps_loss[idx](heatmaps_pred, heatmaps[idx], masks[idx])
                # 將計算出的損失值乘上權重
                heatmaps_loss = heatmaps_loss * self.heatmaps_loss_factor[idx]
                # 將熱力圖損失進行保存
                heatmaps_losses.append(heatmaps_loss)
            else:
                # 如果沒有對應的計算熱力圖損失實例化對象就直接保存None到heatmaps_losses當中
                heatmaps_losses.append(None)

            if self.ae_loss[idx]:
                # 如果當前index有實例化ae損失計算方式就會到這裡
                # 將tags_pred提取出來，會是接續剛才使用完的outputs往下取，tensor shape [batch_size, num_joints, height, width]
                tags_pred = outputs[idx][:, offset_feat:]
                # 獲取batch大小
                batch_size = tags_pred.size()[0]
                # 調整通道順序 [batch_size, num_joints * height * width, 1]
                tags_pred = tags_pred.contiguous().view(batch_size, -1, 1)

                # 將資料放入ae_loss計算損失值
                push_loss, pull_loss = self.ae_loss[idx](tags_pred, joints[idx])
                # 將損失結果乘上權重
                push_loss = push_loss * self.push_loss_factor[idx]
                pull_loss = pull_loss * self.pull_loss_factor[idx]

                # 最後保存起來
                push_losses.append(push_loss)
                pull_losses.append(pull_loss)
            else:
                # 如果沒有實例化對象就直接用None保存
                push_losses.append(None)
                pull_losses.append(None)

        # 回傳損失計算值
        return heatmaps_losses, push_losses, pull_losses
