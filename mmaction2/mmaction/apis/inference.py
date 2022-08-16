# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re
import warnings
from operator import itemgetter

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmaction.core import OutputHook
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_recognizer


def init_recognizer(config, checkpoint=None, device='cuda:0', **kwargs):
    """Initialize a recognizer from config file.

    Args:
        config (str | :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str | None, optional): Checkpoint path/url. If set to None,
            the model will not load any weights. Default: None.
        device (str | :obj:`torch.device`): The desired device of returned
            tensor. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed recognizer.
    """
    # 已看過，透過config資料初始化辨別模型
    # config = 模型的設定資料
    # checkpoint = 訓練權重檔案位置
    # device = 訓練設備
    # kwargs = 其他參數

    if 'use_frames' in kwargs:
        # 如果kwargs當中有說明傳入的資料是frames就會到這裡
        # 會跳出警告表示，目前已經不用強制使用frames作為輸入，可以直接輸入影片就可以
        warnings.warn('The argument `use_frames` is deprecated PR #1191. '
                      'Now you can use models trained with frames or videos '
                      'arbitrarily. ')

    if isinstance(config, str):
        # 如果config是str格式就透過fromfile將config讀取出來
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        # 如果config不是str也不是mmcv.Config就會報錯
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    # pretrained model is unnecessary since we directly load checkpoint later
    # 將config的model當中backbone的預訓練設定成None，因為之後會直接加載整個模型的權重
    config.model.backbone.pretrained = None
    # 構建模型本身
    model = build_recognizer(config.model, test_cfg=config.get('test_cfg'))

    if checkpoint is not None:
        # 如果有傳入訓練好的權重地址，就會在這裡將權重載入
        load_checkpoint(model, checkpoint, map_location='cpu')
    # 將config資料放到model當中
    model.cfg = config
    # 將模型放到設備上
    model.to(device)
    # 將模型設定成驗證模式
    model.eval()
    # 回傳構建好的模型實例對象
    return model


def inference_recognizer(model, video, outputs=None, as_tensor=True, **kwargs):
    """Inference a video with the recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        video (str | dict | ndarray): The video file path / url or the
            rawframes directory path / results dictionary (the input of
            pipeline) / a 4D array T x H x W x 3 (The input video).
        outputs (list(str) | tuple(str) | str | None) : Names of layers whose
            outputs need to be returned, default: None.
        as_tensor (bool): Same as that in ``OutputHook``. Default: True.

    Returns:
        dict[tuple(str, float)]: Top-5 recognition result dict.
        dict[torch.tensor | np.ndarray]:
            Output feature maps from layers specified in `outputs`.
    """
    # 已看過，使用模型進行對影片進行推理
    # model = 模型本身
    # video = 影片的檔案資料位置，或是網址
    # output = 哪些層結構資訊會被返回
    # as_tensor = 像是OutputHook的作用
    if 'use_frames' in kwargs:
        # 如果傳入的是frames就會跳出警告，表示先在可以直接傳入影片就可以
        warnings.warn('The argument `use_frames` is deprecated PR #1191. '
                      'Now you can use models trained with frames or videos '
                      'arbitrarily. ')
    if 'label_path' in kwargs:
        # 現在已經不需要將label_path傳入到inference_recognizer當中
        warnings.warn('The argument `use_frames` is deprecated PR #1191. '
                      'Now the label file is not needed in '
                      'inference_recognizer. ')

    # 先將input_flag設定成None
    input_flag = None
    if isinstance(video, dict):
        # 如果傳入的video是dict格式就會到這裡，將input_flag設定成dict
        input_flag = 'dict'
    elif isinstance(video, np.ndarray):
        # 如果傳入的video是ndarray格式就會到這裡，將input_flag設定成array
        # 檢查一下video的shape是否合法
        assert len(video.shape) == 4, 'The shape should be T x H x W x C'
        input_flag = 'array'
    elif isinstance(video, str) and video.startswith('http'):
        # 如果video是str且是由http開頭的，就將input_flag設定成video
        input_flag = 'video'
    elif isinstance(video, str) and osp.exists(video):
        # 如果是str就會檢查該檔案是否存在
        if osp.isfile(video):
            # 如果是file就會到這裡
            if video.endswith('.npy'):
                # 如果結尾是.npy表示是音樂格式，將input_flag設定成audio
                input_flag = 'audio'
            else:
                # 其他的就是影片格式，將input_flag設定成video
                input_flag = 'video'
        if osp.isdir(video):
            # 如果video是資料夾型態，表示傳入的是frame格式，將input_flag設定成rawframes
            input_flag = 'rawframes'
    else:
        # 其他的就直接報錯
        raise RuntimeError('The type of argument video is not supported: '
                           f'{type(video)}')

    if isinstance(outputs, str):
        # 如果outputs是str，就用tuple進行包裝
        outputs = (outputs, )
    # outputs需要是None或是tuple或是list格式，否則就會報錯
    assert outputs is None or isinstance(outputs, (tuple, list))

    # 獲取構建模型的config資料
    cfg = model.cfg
    # 獲取模型所在的設備
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    # 獲取構建測試資料需要經過的影像處理流水線
    test_pipeline = cfg.data.test.pipeline
    # Alter data pipelines & prepare inputs
    if input_flag == 'dict':
        # 如果input_flag是dict就會到這裡
        data = video
    if input_flag == 'array':
        # 如果input_flag是array就會到這裡
        # 構建一個資料型態的對應，2會對應上光流，3會對應上彩色圖像
        modality_map = {2: 'Flow', 3: 'RGB'}
        # 透過video的shape可以知道該圖像是光流或是彩色圖像
        modality = modality_map.get(video.shape[-1])
        # 構建data的字典
        data = dict(
            # 獲取總共有多少幀
            total_frames=video.shape[0],
            # 因為是測試，所以不會有標註的label，這裡設定成-1
            label=-1,
            # 起始幀
            start_index=0,
            # array就是存放影片本身
            array=video,
            # 將影片的型態輸入
            modality=modality)
        # 遍歷資料處理流的長度
        for i in range(len(test_pipeline)):
            if 'Decode' in test_pipeline[i]['type']:
                # 如果pipeline當中有使用到Decode模塊，這裡就會改成ArrayDecode，因為我們已經將影片處理提取成圖像
                test_pipeline[i] = dict(type='ArrayDecode')
        # 會將帶有Init的pipeline過濾掉
        test_pipeline = [x for x in test_pipeline if 'Init' not in x['type']]
    if input_flag == 'video':
        # 如果input_flag是video就會到這裡
        # 構建data資料，filename設定成影片路徑，label因為是測試模式不會有正確的標註所以是-1，start_index起始的幀，影片格式就是RGB
        data = dict(filename=video, label=-1, start_index=0, modality='RGB')
        if 'Init' not in test_pipeline[0]['type']:
            # 如果pipeline的第一個層結構不是以Init開頭，就會將OpenCVInit作為pipeline的第一個層結構，之後才使原先的層結構
            # 這裡我們改成使用PyAVInit
            test_pipeline = [dict(type='PyAVInit')] + test_pipeline
        else:
            # 如果一開始就有Init，我們強制更換成使用OpenCVInit，這裡使用OpenCV作為基底是因為OpenCV的泛用性較高，隨然OpenCV的性能不好
            # 這裡我們改成PyAVInit
            test_pipeline[0] = dict(type='PyAVInit')
        # 遍歷剩下的pipeline結構
        for i in range(len(test_pipeline)):
            if 'Decode' in test_pipeline[i]['type']:
                # 如果當前的層結構是Decode類的結構就會強制替換成OpenCVDecode，這裡是跟OpenCVInit配套的
                # 這裡我們改成PyAVDecode
                test_pipeline[i] = dict(type='PyAVDecode')
    if input_flag == 'rawframes':
        # 如果Input_flag是rawframes型態就會到這裡
        # 獲取圖像檔案名稱
        filename_tmpl = cfg.data.test.get('filename_tmpl', 'img_{:05}.jpg')
        # 獲取圖像的類型，會是RGB或是Flow，這裡預設會是RGB
        modality = cfg.data.test.get('modality', 'RGB')
        # 開始幀的index
        start_index = cfg.data.test.get('start_index', 1)

        # count the number of frames that match the format of `filename_tmpl`
        # RGB pattern example: img_{:05}.jpg -> ^img_\d+.jpg$
        # Flow patteren example: {}_{:05d}.jpg -> ^x_\d+.jpg$
        # 獲取檔案名稱的模板
        pattern = f'^{filename_tmpl}$'
        if modality == 'Flow':
            # 如果是光流資料就會到這裡
            pattern = pattern.replace('{}', 'x')
        pattern = pattern.replace(
            pattern[pattern.find('{'):pattern.find('}') + 1], '\\d+')
        total_frames = len(
            list(
                filter(lambda x: re.match(pattern, x) is not None,
                       os.listdir(video))))
        data = dict(
            frame_dir=video,
            total_frames=total_frames,
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality)
        if 'Init' in test_pipeline[0]['type']:
            test_pipeline = test_pipeline[1:]
        for i in range(len(test_pipeline)):
            if 'Decode' in test_pipeline[i]['type']:
                test_pipeline[i] = dict(type='RawFrameDecode')
    if input_flag == 'audio':
        # 如果input_flag是audio就會到這裡
        data = dict(
            audio_path=video,
            total_frames=len(np.load(video)),
            start_index=cfg.data.test.get('start_index', 1),
            label=-1)

    # 構建影片處理流水線
    test_pipeline = Compose(test_pipeline)
    # 將data放入到流水線當中處理
    data = test_pipeline(data)
    # 透過collate將資料整理成一個batch
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # 如果有使用gpu就會到這裡
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with OutputHook(model, outputs=outputs, as_tensor=as_tensor) as h:
        # 在OutputHook下
        with torch.no_grad():
            # 將模型的反向傳遞關閉，進行正向傳遞
            scores = model(return_loss=False, **data)[0]
        # 如果有需要獲取層結構輸出就會到這裡獲取
        returned_features = h.layer_outputs if outputs else None

    # 獲取總共是多少分類
    num_classes = scores.shape[-1]
    # 將一個類別對應上一個置信度分數，score_tuples = tuple(tuple(對應類別的index, 置信度分數))，第一個tuple長度就會是num_classes
    score_tuples = tuple(zip(range(num_classes), scores))
    # 將score_tuples進行排序，這裡會從置信度大排到小
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)

    # 獲取前5大概率值的置信度以及分類類別
    top5_label = score_sorted[:5]
    if outputs:
        # 如果有需要中途的輸出就會到這裡
        return top5_label, returned_features
    # 其他就會輸出前5大預測結果
    return top5_label
