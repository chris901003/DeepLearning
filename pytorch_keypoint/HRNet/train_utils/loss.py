import torch


class KpLoss(object):
    def __init__(self):
        # 已看過
        # MSELoss = 均方差損失計算(x - y) ^ 2
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        """
        :param logits: 模型預測輸出shape [batch_size, 關節點數量, height, width]
        :param targets: 真實標籤訊息
        :return:
        """
        # 已看過
        # 檢查logits是否符合標準
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        # 檢查訓練設備
        device = logits.device
        # 獲取batch_size
        bs = logits.shape[0]
        # 把target中的heatmap堆疊起來，就會在最前面多出batch_size維度
        # [num_kps, height, width] -> [batch_size, num_kps, height, width]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # 也將權重部分進行堆疊
        # [num_kps] -> [batch_size, num_kps]
        kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])

        # [batch_size, num_kps, height, width] -> [batch_size, num_kps]
        # 將正確輸出與預測輸出拿去進行損失計算，這裡我們指定在維度2與3上面計算
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        # 將計算出來的損失乘上個關節點的權重，最後除以batch_size
        loss = torch.sum(loss * kps_weights) / bs
        return loss
