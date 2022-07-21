# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    # 已看過，我們會在預測時使用到這裡
    # config = 模型配置文件，可以是檔案位置或是已經解析過的mmcv.Config類型
    # checkpoint = 預訓練權重資訊，如果沒有放就不會載入任何權重
    # device = 要在哪個設備上運行

    if isinstance(config, str):
        # 如果config傳入的是檔案位置就透過fromfile進行解析
        # 這部分與train在解析config文件時相同
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        # 如果不是str就要是mmcv.Config格式否則就會報錯
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    # 將模型當中的pretrained與train_cfg設定成None
    config.model.pretrained = None
    config.model.train_cfg = None
    # 透過build_segmentor進行模型構建
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        # 如果指定訓練權重就在這裡進行載入
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # 將CLASSES與PALETTE傳給model當中
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    # 將模型調整到設備上
    model.to(device)
    # 將模型調整到驗證模式
    model.eval()
    # 回傳模型
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        # 已看過，在測試當中會是讀取圖像的第一步
        # results = dict{img: str}

        if isinstance(results['img'], str):
            # 保存檔案名稱
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            # 如果傳入的不是str就將當案名稱預設為None
            results['filename'] = None
            results['ori_filename'] = None
        # 透過imread進行圖像讀取
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_segmentor(model, img):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    # 已看過，會將一張圖像的檔案路徑傳入進行預測
    # model = 模型本身
    # img = 圖像檔案路徑

    # cfg = 構建模型時的配置
    cfg = model.cfg
    # 獲取驗證的設備
    device = next(model.parameters()).device  # model device
    # build the data pipeline，構建獲取資料的流
    # 這裡第一步我們改用這裡實現的LoadImage函數其他就直接用config當中test的流
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    # 將獲得的圖像獲取流放入到Compose當中進行實例化
    test_pipeline = Compose(test_pipeline)
    # prepare data，準備資料
    data = dict(img=img)
    # 將圖像放到資料處理流當中
    data = test_pipeline(data)
    # data裡面就會有處理好的圖像資料，這裡已經將圖像轉成tensor格式
    # 透過collate就可以獲取一個batch的格式，這個也就是collect_fn
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model，進行模型的正向傳遞
    with torch.no_grad():
        # result = list[ndarray]，ndarray shape [height, width]，list長度就會是batch_size，這裡通常都是1
        result = model(return_loss=False, rescale=True, **data)
    return result


def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True,
                       out_file=None):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    # 已看過，用來將預測出來的mask放到圖像上面
    # model = 模型本身
    # img = 原始圖像檔案位置
    # result = 透過模型預測出來的mask，高寬會與原始圖像相同
    # palette = 用來調色用的，表示每一種類別要用哪種顏色
    # fig_size = pyplot的圖像大小
    # opacity = 不透明度
    # title = pyplot的標題
    # block = 是否啟用pyplot
    # out_file = 輸出圖像

    if hasattr(model, 'module'):
        # 在DDP模式下model外面會多包一層，這裡會進行脫層
        model = model.module
    # 透過model當中的show_result方式或取圖像
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
