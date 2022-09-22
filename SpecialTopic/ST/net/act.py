from torch import nn
import torch


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        module = SiLU()
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError(f'Unsupported act type {name}')
    return module


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
