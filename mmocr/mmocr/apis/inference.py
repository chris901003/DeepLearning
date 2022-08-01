# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from mmocr.models import build_detector
from mmocr.utils import is_2dlist
from .utils import disable_text_recog_aug_test


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    # 已看過，構建偵測模型並且同時進行初始化以及權重加載
    # config = 指定模型config文件檔案路徑
    # checkpoint = 模型權重檔案位置
    # device = 運行設備
    # cfg_options = config額外設定或是將原始config進行複寫
    if isinstance(config, str):
        # 如果config是檔案位置就會透過fromfile讀取config文件內容
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        # 如果傳入的config是mmcv.Config就直接使用，否則就會報錯
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        # 如果有需要複寫或是增加config文件內容的就到這裡
        config.merge_from_dict(cfg_options)
    if config.model.get('pretrained'):
        # 如果config文件當中有預訓練權重加載設定就先設定成None，因為會等等會加載整個模型權重
        config.model.pretrained = None
    # 將train_cfg設定成None
    config.model.train_cfg = None
    # 構建預測模型，獲取模型實例化對象
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        # 如果有傳入訓練權重資料就會進來
        # 透過load_checkpoint進行權重加載
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            # 如果在權重當中的meta有找到CLASSES就會使用
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            # 否則就直接使用coco的80分類的CLASSES
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    # 將config放到model當中
    model.cfg = config  # save the config in the model for convenience
    # 將模型放到設備當中
    model.to(device)
    # 將模型設定成驗證模式
    model.eval()
    # 將模型回傳
    return model


def model_inference(model,
                    imgs,
                    ann=None,
                    batch_mode=False,
                    return_data=False):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.
        batch_mode (bool): If True, use batch mode for inference.
        ann (dict): Annotation info for key information extraction.
        return_data: Return postprocessed data.
    Returns:
        result (dict): Predicted results.
    """
    # 已看過，模型的介面
    # model = 模型本身
    # imgs = 輸入圖像，會是str或是ndarray，如果是ndarray就是已經經過讀取的圖像
    # ann = 標註訊息，如過是在測試就會是None
    # batch_mode = 是否啟用batch模式
    # return_data = 是否回傳後處理資料

    if isinstance(imgs, (list, tuple)):
        # 如果傳入的imgs是list或是tuple格式就會到這裡
        # 先將is_batch設定成True
        is_batch = True
        if len(imgs) == 0:
            # 如果傳入的圖像數量是0就會報錯，表示沒有圖像
            raise Exception('empty imgs provided, please check and try again')
        if not isinstance(imgs[0], (np.ndarray, str)):
            # 檢查當中資料需要是str或是ndarray格式
            raise AssertionError('imgs must be strings or numpy arrays')

    elif isinstance(imgs, (np.ndarray, str)):
        # 如果傳入的是ndarray或是str就在外面加上list並且將is_batch設定成False
        imgs = [imgs]
        is_batch = False
    else:
        # 其他情況就會直接報錯
        raise AssertionError('imgs must be strings or numpy arrays')

    # 檢查imgs當中資料是否為ndarray格式，如果是str就會需要將圖像讀取出來
    is_ndarray = isinstance(imgs[0], np.ndarray)

    # 獲取模型的config資料
    cfg = model.cfg

    if batch_mode:
        # 如果有開啟batch模式就會到這裡
        cfg = disable_text_recog_aug_test(cfg, set_types=['test'])

    # 獲取模型所在的設備
    device = next(model.parameters()).device  # model device

    if cfg.data.test.get('pipeline', None) is None:
        # 如果沒有找到config當中圖像處理的pipeline就會到這裡
        if is_2dlist(cfg.data.test.datasets):
            cfg.data.test.pipeline = cfg.data.test.datasets[0][0].pipeline
        else:
            cfg.data.test.pipeline = cfg.data.test.datasets[0].pipeline
    if is_2dlist(cfg.data.test.pipeline):
        # 如果當中是list[list]結構就會將第一層的list去除
        cfg.data.test.pipeline = cfg.data.test.pipeline[0]

    if is_ndarray:
        # 如果傳入的資料是ndarray格式就會到這裡
        cfg = cfg.copy()
        # set loading pipeline type
        # 將原先第一步驟的讀取圖像從檔案位置更改成讀取圖像從ndarray
        cfg.data.test.pipeline[0].type = 'LoadImageFromNdarray'

    # 將ImageToTensor轉成DefaultFormatBundle
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    # 透過Compose將pipeline實例化
    test_pipeline = Compose(cfg.data.test.pipeline)

    # 最終data保存的位置
    datas = []
    # 遍歷所有傳入的圖像
    for img in imgs:
        # prepare data
        if is_ndarray:
            # 如果傳入的是ndarray型態就會到這裡
            # directly add img，構建data的dict
            data = dict(
                img=img,
                ann_info=ann,
                img_info=dict(width=img.shape[1], height=img.shape[0]),
                bbox_fields=[])
        else:
            # 如果傳入的圖像檔案位置
            # add information into dict
            data = dict(
                img_info=dict(filename=img),
                img_prefix=None,
                ann_info=ann,
                bbox_fields=[])
        if ann is not None:
            # 如果有傳入標注訊息就會到這裡
            data.update(dict(**ann))

        # build the data pipeline
        # 將data的dict放入到pipeline當中進行一系列處理，data就會是處理好的一個batch的資料
        data = test_pipeline(data)
        # get tensor from list to stack for batch mode (text detection)
        if batch_mode:
            # 如果使用batch模式就會到這裡
            if cfg.data.test.pipeline[1].type == 'MultiScaleFlipAug':
                # 將pipeline第二個步驟是MultiScaleFlipAug
                for key, value in data.items():
                    data[key] = value[0]
        datas.append(data)

    if isinstance(datas[0]['img'], list) and len(datas) > 1:
        raise Exception('aug test does not support '
                        f'inference with batch size '
                        f'{len(datas)}')

    # 透過collate將datas資料統整，會將圖像tensor變成shape [batch_size, channel, height, width]
    data = collate(datas, samples_per_gpu=len(imgs))

    # process img_metas
    if isinstance(data['img_metas'], list):
        # 更新img_metas當中的資料
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
    else:
        data['img_metas'] = data['img_metas'].data

    if isinstance(data['img'], list):
        # 將data中的img不要的資訊去除，最後留下一個batch的圖像資料
        data['img'] = [img.data for img in data['img']]
        if isinstance(data['img'][0], list):
            data['img'] = [img[0] for img in data['img']]
    else:
        data['img'] = data['img'].data

    # for KIE models
    if ann is not None:
        # 如果有給定ann資料就會進來
        data['relations'] = data['relations'].data[0]
        data['gt_bboxes'] = data['gt_bboxes'].data[0]
        data['texts'] = data['texts'].data[0]
        data['img'] = data['img'][0]
        data['img_metas'] = data['img_metas'][0]

    if next(model.parameters()).is_cuda:
        # 如果是用gpu推力就會到這裡
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            # 這裡如果有用到RoIPool就會報錯，因會非gpu的還沒有實現
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model，將模型的反向傳遞關閉，進行模型推理
    with torch.no_grad():
        # result = list[dict]，list長度就會是一個batch的圖像數量
        # dict = {
        #   'filename': 檔案名稱
        #   'boundary_result': list[list]，第一個list長度就會是有多少個匡，第二個list就是匡的詳細資訊
        # }
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        # 如果沒有使用batch模式
        if not return_data:
            #  如果不需要將細節回傳就直接回傳result內容
            return results[0]
        # 否則就要連datas也需要回傳
        return results[0], datas[0]
    else:
        # 使用batch模式就會到這裡
        if not return_data:
            return results
        return results, datas


def text_model_inference(model, input_sentence):
    """Inference text(s) with the entity recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        input_sentence (str): A text entered by the user.

    Returns:
        result (dict): Predicted results.
    """

    assert isinstance(input_sentence, str)

    cfg = model.cfg
    if cfg.data.test.get('pipeline', None) is None:
        if is_2dlist(cfg.data.test.datasets):
            cfg.data.test.pipeline = cfg.data.test.datasets[0][0].pipeline
        else:
            cfg.data.test.pipeline = cfg.data.test.datasets[0].pipeline
    if is_2dlist(cfg.data.test.pipeline):
        cfg.data.test.pipeline = cfg.data.test.pipeline[0]
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = {'text': input_sentence, 'label': {}}

    # build the data pipeline
    data = test_pipeline(data)
    if isinstance(data['img_metas'], dict):
        img_metas = data['img_metas']
    else:
        img_metas = data['img_metas'].data

    assert isinstance(img_metas, dict)
    img_metas = {
        'input_ids': img_metas['input_ids'].unsqueeze(0),
        'attention_masks': img_metas['attention_masks'].unsqueeze(0),
        'token_type_ids': img_metas['token_type_ids'].unsqueeze(0),
        'labels': img_metas['labels'].unsqueeze(0)
    }
    # forward the model
    with torch.no_grad():
        result = model(None, img_metas, return_loss=False)
    return result
