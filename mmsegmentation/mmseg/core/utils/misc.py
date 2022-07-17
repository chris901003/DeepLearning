# Copyright (c) OpenMMLab. All rights reserved.
def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """
    # 已看過
    # 更改dict當中的key使用的，透過添加prefix可以區別從哪裡輸出的

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs
