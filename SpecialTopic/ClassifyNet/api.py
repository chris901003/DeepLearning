import torch
import numpy as np
from SpecialTopic.ST.build import build_detector
from SpecialTopic.ST.dataset.utils import Compose


def init_model(cfg='none', model_type='ResNet', phi='m', num_classes=100, device='auto', pretrained='none',
               with_pipeline=True, input_size=(224, 224)):
    if device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_cfg = {
        # 這裡需要提供使用哪種模型
        'type': model_type,
        # 提供模型尺寸
        'phi': phi,
        # 提供類別數量
        'num_classes': num_classes,
        # 這裡會將骨幹的預訓練權重設定成none，之後會對整個模型加載權重
        'pretrained': 'none'
    }
    if cfg != 'none':
        # 如果想要完全自定義模型設定檔，就直接把整個設定檔傳入
        model_cfg = cfg
    model = build_detector(model_cfg)
    if pretrained != 'none':
        pretrained_dict = torch.load(pretrained, map_location='cpu')
        if 'model_weight' in pretrained_dict:
            model_weight = pretrained_dict['model_weight']
        else:
            model_weight = pretrained_dict
        model.load_state_dict(model_weight)
    else:
        print('未加載訓練好權重，預測是無效的')
    model = model.to(device)
    model = model.eval()
    if with_pipeline:
        pipeline_cfg = [
            {'type': 'ResizeSingle', 'input_shape': input_size, 'save_info': False, 'keep_ratio': True},
            {'type': 'NormalizeSingle', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'to_rgb': True},
            {'type': 'Collect', 'keys': ['image']}]
        pipeline = Compose(pipeline_cfg)
        model.pipeline = pipeline
    return model


def detect_single_picture(model, image, pipeline=None, device='auto'):
    assert isinstance(image, np.ndarray)
    if device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = dict(image=image, label=0)
    if pipeline is not None:
        data = pipeline(data)
    elif hasattr(model, 'pipeline'):
        data = model.pipeline(data)
    else:
        raise RuntimeError('沒有提供資料處理流')
    image = data['image']
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    image = image.unsqueeze(dim=0)
    image = image.to(device)
    model = model.to(device)
    output = model(image, with_loss=False)
    return output
