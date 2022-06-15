from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist

import errno
import os


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        """
        :param window_size: deque的大小
        :param fmt: 輸出的格式
        """
        # 已看過
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        # 構建deque並且限制長度
        self.deque = deque(maxlen=window_size)
        # 一些記錄用的，最後可以計算平均之類的
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        # 已看過
        # 更新資料
        # n = 權重，基本上都是1
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        # 已看過
        # 計算平均值
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class ConfusionMatrix(object):
    # 混淆矩陣，由evaluate構建
    def __init__(self, num_classes):
        # 已看過
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        # a, b shape [batch_size * w * h]
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            # 第一次的時候會需要構建矩陣
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            # 過濾掉不好分別的部分，像是物體邊緣(這些不好分別的我們都是用255代表所以這裡會過濾掉)
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            # 找到在混淆矩陣上面要填的位置
            inds = n * a[k].to(torch.int64) + b[k]
            # 計算inds裡數字出現的次數紀錄在一個n*n的矩陣上，最後再與mat相加，就完成更新
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        # 已看過
        # 將混淆矩陣轉成float格式
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        # 已看過
        # 多gpu時需要同步
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        # 已看過
        # 將內容轉換成string格式
        # acc_global = 全局的正確率
        # acc = 每一個分類的正確率
        # iu = 每一個分類的iou
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        # 已看過
        # meters型態dict，value預設為SmoothedValue型態
        self.meters = defaultdict(SmoothedValue)
        # delimiter每個資訊之間用兩個空白
        self.delimiter = delimiter

    def update(self, **kwargs):
        # 已看過
        # 更新資料
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        # 已看過
        # 遍歷存在meter中的所有key與value並且變成string輸出出去
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        # 已看過
        # 在meters中構建一個新的key與value
        self.meters[name] = meter

    # 讀取一個batch的資料
    def log_every(self, iterable, print_freq, header=None):
        """
        :param iterable: dataloader
        :param print_freq: 多少個batch會打印一次
        :param header: 目前是第幾個epoch
        :return:
        """
        # 紀錄當前是在第幾個batch
        i = 0
        if not header:
            header = ''
        # 紀錄時間
        start_time = time.time()
        end = time.time()
        # 預測一個batch所花的平均時間
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        # 讀取一個batch資料所花的平均時間
        data_time = SmoothedValue(fmt='{avg:.4f}')
        # 讓顯示當前是第幾個batch的長度與總共batch長度要相同，在顯示的時候才會比較整齊
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        # 有gpu與沒有gpu差別只有max mem，max mem是在紀錄訓練過程中gpu最大吃了多少ram
        # 添加上要輸出的資訊，花括號裡面的之後會放上對應的變數
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        # 遍歷一整個dataloader
        for obj in iterable:
            # 計算出一個batch資料讀取所花的時間
            data_time.update(time.time() - end)
            # yield是python的一種return方式，只是可以記錄下當前狀態，下次再呼叫log_every時會從這行的下行開始執行
            # 這樣可以節省記憶體空間
            yield obj
            # 記錄下一個batch預測所需要的時間
            iter_time.update(time.time() - end)
            # 看是否需要進行打印
            if i % print_freq == 0:
                # 透過計算一個batch處理需要的時間，可以知道還需要多少時間可以完成一個epoch
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                # 將秒數轉成時分秒的格式，同時也將型態轉成string
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                # 打印一些訊息，gpu與沒有gpu差別在memory，gpu記憶體使用量
                # 將變數填入進去，meters的self可以到上面的__str__看
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            # 當前batch加一
            i += 1
            # 結束時間
            end = time.time()
        # 一個epoch總花費時間
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # 輸出一個epoch總花費時間
        print('{} Total time: {}'.format(header, total_time_str))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
