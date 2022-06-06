# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from packaging import version
from typing import Optional, List

import torch
import torch.distributed as dist
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
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


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        # 單gpu裝態
        return input_dict
    with torch.no_grad():
        # 等到時候看input_dict裡面有什麼再來說
        # names = dict的key, values = dict的value
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        # 對輸入的dict的key做sort，為了讓每個gpu出來的dict的key順序是一樣的
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        # list轉換成tensor
        values = torch.stack(values, dim=0)
        # 官方實現的每個gpu做加總
        dist.all_reduce(values)
        if average:
            values /= world_size
        # 再放回到dict裡面返回
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
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
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
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
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
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
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    # 已看過
    # 一個batch的訓練資料都會到這裡來變成一個可堆疊的batch
    # 一開始的圖片大小都不一樣沒辦法直接堆一起
    # 記得要轉成list否則會變成zip格式沒辦法用
    batch = list(zip(*batch))
    # 取出一個batch的照片進行處理，這裡的image已經是tensor格式了
    # batch[0]變成NestedTensor的格式了，也就是圖片的tensor以及mask封裝在一個class裡面
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    # 已看過
    # the_list = 每張圖片的shape (List(List))
    maxes = the_list[0]
    # 這個batch中，每個維度上面的最大
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        # 已看過
        # 由下面的nested_tensor_from_tensor_list實例化
        # tensor shape [batch_size, 3, w, h]
        # mask shape [batch_size, w, h]
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        # 已看過
        # 調用to方法，轉換設備
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        # 已看過
        # 分別輸出tensors以及mask
        return self.tensors, self.mask

    def __repr__(self):
        # 已看過
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    # 已看過
    # 傳入的是已經轉成tensor格式的多個image組成的list
    # 我們知道一張圖片轉成tensor後會shape=[3, w, h]，ndim就是可以看嵌套層數這裡就會是3
    # 這裡就是在過濾不合法的
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # onnx在用的，有機會再來研究onnx怎麼搞
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        # 把每張圖片的shape傳入_max_by_axis
        # 找到這個batch中每個維度的最大[3, max_w, max_h]
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        # batch_shape shape = [batch_size, 3, max_w, max_h]
        batch_shape = [len(tensor_list)] + max_size
        # 最後都會把照片的tensor的高寬維度調整成跟最大一樣
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        # 構建一個與batch_shape一樣shape但初始值為0的tensor
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        # 一個shape=[b, h, w]且全為1的mask，這個mask不需要channel維度因為他只記錄這個點是不是真得有圖就可以了
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        # 開始把每個tensor堆疊在一起
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            # 在新的tensor上從最左上角開始貼上，貼不完的部分就是0
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            # 記錄下每張圖那些部分是後來填充的
            # 不足的部分會是True，真實有圖的地方會是False
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    # tensor shape [batch_size, 3, w, h]
    # mask shape [batch_size, w, h]
    # 轉換成NestedTensor格式
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    # 已看過
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    # 已看過
    # 看是不是在多gpu上運行
    if not dist.is_available():
        # Returns True if the distributed package is available.
        return False
    if not dist.is_initialized():
        # Checking if the default process group has been initialized
        return False
    return True


def get_world_size():
    # 已看過
    # 回傳有多少塊gpu
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
    # 已看過
    # 在多gpu下我們只保存主線程上的
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # 已看過
    # 在這個腳本中使用多gpu訓練時用指令是:torch.distributed.launch
    # 在調用指令時會有--use_env，這個指令會在os.environ裡面放入RANK, WORLD_SIZE, LOCAL_RANK變數
    # ---------------------------------------------------------
    # 多機多卡時，WORD_SIZE對應所有機器中使用的進程數量(一個進程對應一塊gpu)
    # RANK代表所有進程中的第幾個進程，LOCAL_RANK對應當前機器中第幾個進程
    # 在單機多卡下
    # WORD_SIZE就是這台設備上有多少gpu，LOCAL_RANK與RANK表示都是當前是哪塊gpu
    # ---------------------------------------------------------
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    # 表示使用多gpu，在main中有用到這個變數
    args.distributed = True

    # 指定當前gpu
    torch.cuda.set_device(args.gpu)
    # 通信後端，nvidia GPU推薦使用nccl
    args.dist_backend = 'nccl'
    # 打印一些關於多gpu的信息
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    # 很重要的部分，建立進程組
    # ---------------------------------------------------------
    # backend = 剛剛設定的通訊後端
    # init_method = 初始化方法，這邊都是用默認方法
    # ---------------------------------------------------------
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    # 等待所有gpu都結束上面的操作
    torch.distributed.barrier()
    # This function disables printing when not in master process
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # 已看過
    # src_logits[idx] = 有對應上gt_box的query對於每個分類類別的預測值
    # target_classes_o = 正確分類類別
    # src_logits shape [total_number_match_gt_box, num_classes]
    # target_classes_o shape [total_number_match_gt_box]
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    # 對於output找前1大(小)的值且在維度1上尋找，第一個True表示找前k大，第二個True表示返回時需要排序
    # 返回值為value, index
    # pred shape [total_number_match_gt_box, 1]
    _, pred = output.topk(maxk, 1, True, True)
    # 轉置 pred shape [total_number_match_gt_box, 1] -> [1, total_number_match_gt_box, 1]
    pred = pred.t()
    # 正常來說只需要在前面擴圍就可以了，後面的expand_as基本不起作用
    # correct shape [1, total_number_match_gt_box]相同的地方會是True，不同的地方是False
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    # 遍歷我們要的topk，這裡我們只要top1就可以了
    for k in topk:
        # correct[:k] = [total_number_match_gt_box]
        # 看總共有多少個True
        correct_k = correct[:k].view(-1).float().sum(0)
        # batch_size表示有多少個gt_box，答對數量除以總數量乘以100可以得到%數
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)
