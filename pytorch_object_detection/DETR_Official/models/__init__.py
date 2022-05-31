# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build


def build_model(args):
    # 已看過
    return build(args)
