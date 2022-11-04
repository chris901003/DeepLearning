import torch
import os
import numpy as np


def load_pretrained(model, pretrained_path):
    assert os.path.exists(pretrained_path), '需提供預訓練權重資料'
    print(f'Load weights {pretrained_path}')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    if 'model_weight' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_weight']
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        idx = k.find('.')
        new_name = k[idx + 1:]
        if new_name in model_dict.keys() and np.shape(model_dict[new_name]) == np.shape(v):
            temp_dict[new_name] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    assert len(no_load_key) == 0, '給定的預訓練權重與模型不匹配'
    return model
