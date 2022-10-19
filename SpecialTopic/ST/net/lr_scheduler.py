import math
from functools import partial
import torch
import copy


def get_lr_scheduler_yolox(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.5, warmup_lr_ratio=0.1,
                           no_aug_iter_ratio=0.5, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 +
                    math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return func


def set_optimizer_lr_yolox(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_lr_scheduler(optimizer, lr_scheduler_cfg):
    """
    Args:
         optimizer: 優化器本身，如果要使用lr schedule需要提供當前使用的優化器
         lr_scheduler_cfg: 學習率初始化設定資料
    """
    # 當前支援的學習率調整方式
    support_lr_scheduler = {
        # 這裡的type就會是torch當中的lr_schedule的其中一種方式
        'cos_lr_scheduler': {'type': torch.optim.lr_scheduler.LambdaLR, 'lambda': cos_lr_scheduler},
        'StepLR': {'type': torch.optim.lr_scheduler.StepLR},
        'MultiStepLR': {'type': torch.optim.lr_scheduler.MultiStepLR},
        'ExponentialLR': {'type': torch.optim.lr_scheduler.ExponentialLR},
        'CosineAnnealingLR': {'type': torch.optim.lr_scheduler.CosineAnnealingLR},
        # 這個學習率調整方式比較特別，如果不會使用請看官方文檔
        'ReduceLROnPlateau': {'type': torch.optim.lr_scheduler.ReduceLROnPlateau}
    }
    lr_scheduler_cfg_ = copy.deepcopy(lr_scheduler_cfg)
    lr_scheduler_type = lr_scheduler_cfg_.pop('type', None)
    assert lr_scheduler_type is not None, '需提供指定的lr scheduler類型(type標籤)'
    lr_scheduler_info = support_lr_scheduler.get(lr_scheduler_type, None)
    assert lr_scheduler_info is not None, f'目前尚未支持{lr_scheduler_type}類型，如果有需要請自行撰寫'
    lr_scheduler_api = lr_scheduler_info.get('type', None)
    assert lr_scheduler_api, '需要提供更新學習率的api'
    lambda_cfg = lr_scheduler_cfg_.pop('lambda_cfg', None)
    if 'lambda' in lr_scheduler_info.keys():
        # 獨立將需要帶入lambda的到這裡
        func = lr_scheduler_api(optimizer=optimizer, **lr_scheduler_cfg_,
                                lr_lambda=lr_scheduler_info['lambda'](**lambda_cfg))
    else:
        # 其他就直接到這裡將參數放入
        func = lr_scheduler_api(optimizer=optimizer, **lr_scheduler_cfg_)
    return func


def cos_lr_scheduler(lr, min_lr, total_iters, warmup_iters_ratio=0.5, warmup_lr_ratio=0.1, no_aug_iter_ratio=0.5):
    """ cos學習率下降方式
    Args:
        lr: 最大學習率值，也就是最開始設定的lr值
        min_lr: 最小學習率
        total_iters: 總共需要訓練多少個Epoch
        warmup_iters_ratio: 熱身訓練的Epoch比例，但是也不會超過指定數量
        warmup_lr_ratio: 在熱身訓練時的lr會是原始最大學習率的多少倍
        no_aug_iter_ratio: 沒有輔助的Epoch比例
    """

    def warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        ori_lr = lr
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 +
                    math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr / ori_lr
    warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
    warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
    no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
    func = partial(warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    return func


def test():
    import matplotlib.pyplot as plt

    # 提供測試使用，可以在撰寫新的學習率下降後進行測試
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc = torch.nn.Linear(3, 20)
    net = Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    # 如果有設定last_epoch就會需要，主要是中斷訓練後繼續訓練需要的東西
    optimizer.param_groups[0]['initial_lr'] = 1e-2
    lr_scheduler_cfg = {
        'type': 'CosineAnnealingLR', 'T_max': 5, 'last_epoch': -1
    }
    lr_scheduler = build_lr_scheduler(optimizer, lr_scheduler_cfg)
    x, y = list(), list()
    for i in range(25):
        lr_scheduler.step()
        x.append(i)
        y.append(optimizer.state_dict()['param_groups'][0]['lr'])
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    print('Testing lr scheduler')
    test()
