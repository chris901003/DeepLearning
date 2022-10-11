import torch
from torch import nn
from functools import partial
import math


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc = torch.nn.Linear(3, 10)

    def forward(self, x):
        x = self.fc(x)
        return x


def test_lambda(epoch):
    return 1e-2 * (0.1 ** epoch)


def cos_lr_scheduler(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
    ori_lr = lr
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
    elif iters >= total_iters - no_aug_iter:
        lr = min_lr
    else:
        lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 +
                math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)))
    return lr / ori_lr


model = Net()
pg0, pg1 = list(), list()
for k, v in model.named_modules():
    if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
        pg0.append(v.bias)
    if hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
        pg1.append(v.weight)
optimizer = torch.optim.SGD(pg0, lr=0.1)
lr_test_lambda = partial(test_lambda)
lr_cos_lambda = partial(cos_lr_scheduler, 1e-2, 1e-4, 100, 3, 1e-5, 10)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_cos_lambda, last_epoch=-1)

print(0, scheduler.get_lr())
for epoch in range(100):
    scheduler.step()
    print(epoch, scheduler.get_lr())
