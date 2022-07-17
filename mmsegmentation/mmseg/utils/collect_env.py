# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmseg


def collect_env():
    """Collect the information of the running environments."""
    # 已看過
    # 主要用來收集當前環境的資訊
    env_info = collect_base_env()
    # 在資訊當中添加上MMSegmentation的版本資訊
    env_info['MMSegmentation'] = f'{mmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


if __name__ == '__main__':
    # 這個是測試用的
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
