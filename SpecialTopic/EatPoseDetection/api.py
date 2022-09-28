import torch
from SpecialTopic.ST.build import build_detector


def init_recognizer(cfg=None, num_classes=None, checkpoint=None, device=None):
    if cfg is None and num_classes is None:
        raise ValueError('至少需要提供完整模型config或是指定分類類別數')
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_cfg = {
        'type': 'Recognizer3D',
        'backbone': {
            'type': 'ResNet3d',
            'pretrained2d': False,
            'pretrained': None,
            'depth': 50,
            'conv1_kernel': (5, 7, 7),
            'conv1_stride_t': 2,
            'pool1_stride_t': 2,
            'conv_cfg': {'type': 'Conv3d'},
            'norm_eval': False,
            'inflate': ((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
            'zero_init_residual': False
        },
        'cls_head': {
            'type': 'I3DHead',
            'num_classes': num_classes,
            'in_channels': 2048,
            'spatial_type': 'avg',
            'dropout_ratio': 0.5,
            'init_std': 0.01
        },
        'train_cfg': {},
        'test_cfg': {
            'average_clips': 'prob'
        }
    }
    if cfg is not None:
        model_cfg = cfg
    model = build_detector(model_cfg)
    if checkpoint is not None and checkpoint != 'none':
        pretrained_dict = torch.load(checkpoint, map_location=device)
        if 'model_state_dict' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['model_state_dict']
        model.load_state_dict(pretrained_dict)
    else:
        print('建議加載預訓練權重，不然就只是在預測爽的')
    model.to(device)
    model.eval()
    return model
