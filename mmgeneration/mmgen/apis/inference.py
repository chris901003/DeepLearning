# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.utils import is_list_of

from mmgen.datasets.pipelines import Compose
from mmgen.models import BaseTranslationModel, build_model


def init_model(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Pretrained weight
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed unconditional model.
    """
    """ 初始化模型
    Args:
        config: 模型設定資料config文件
        checkpoint: 預訓練權重資料
        device: 運行設備
        cfg_options: 額外添加或是要更新到config資料當中的資料
    """

    if isinstance(config, str):
        # 將config資料進行讀入，如果給定的config是檔案路徑
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        # 否則就會需要是mmcv的config格式
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        # 如果有額外的config資料就會到這裡進行融合
        config.merge_from_dict(cfg_options)

    # 構建模型
    model = build_model(config.model, train_cfg=config.train_cfg, test_cfg=config.test_cfg)

    if checkpoint is not None:
        # 將預訓練權重進行載入
        load_checkpoint(model, checkpoint, map_location='cpu')

    # 將config資料保存一份到model當中
    model._cfg = config  # save the config in the model for convenience
    # 將模型放到指定設備上
    model.to(device)
    # 將模型轉成驗證模式
    model.eval()

    # 回傳構建好的模型實例化對象
    return model


@torch.no_grad()
def sample_unconditional_model(model,
                               num_samples=16,
                               num_batches=4,
                               sample_model='ema',
                               **kwargs):
    """Sampling from unconditional models.

    Args:
        model (nn.Module): Unconditional models in MMGeneration.
        num_samples (int, optional): The total number of samples.
            Defaults to 16.
        num_batches (int, optional): The number of batch size for inference.
            Defaults to 4.
        sample_model (str, optional): Which model you want to use. ['ema',
            'orig']. Defaults to 'ema'.

    Returns:
        Tensor: Generated image tensor.
    """
    """ 對於沒有條件的模型進行創建圖像
    Args:
        model: 模型實例化對象
        num_samples: 總共需要產生多少張圖像
        num_batches: 每一次會產生多少張圖像
        sample_model: 最一開始的噪聲產生方式
    """
    # set eval mode
    # 將模型轉成驗證模式
    model.eval()
    # construct sampling list for batches
    # 計算總共需要生成多少次
    n_repeat = num_samples // num_batches
    # 構建batches_list，在生成的時候就會根據這個決定一個batch的大小
    batches_list = [num_batches] * n_repeat

    if num_samples % num_batches > 0:
        # 如果有少的話就會添加上去
        batches_list.append(num_samples % num_batches)
    # 最終結果的list
    res_list = []

    # inference
    # 遍歷需要生成的batch大小
    for batches in batches_list:
        # 進行生成
        res = model.sample_from_noise(None, num_batches=batches, sample_model=sample_model, **kwargs)
        # 將生成結果轉到cpu上並且保存到res_list當中
        res_list.append(res.cpu())

    # 將結果沿著dim=0進行拼接
    results = torch.cat(res_list, dim=0)

    # 最後回傳結果，tensor shape [num_sample, channel, height, width]
    return results


@torch.no_grad()
def sample_conditional_model(model,
                             num_samples=16,
                             num_batches=4,
                             sample_model='ema',
                             label=None,
                             **kwargs):
    """Sampling from conditional models.

    Args:
        model (nn.Module): Conditional models in MMGeneration.
        num_samples (int, optional): The total number of samples.
            Defaults to 16.
        num_batches (int, optional): The number of batch size for inference.
            Defaults to 4.
        sample_model (str, optional): Which model you want to use. ['ema',
            'orig']. Defaults to 'ema'.
        label (int | torch.Tensor | list[int], optional): Labels used to
            generate images. Default to None.,

    Returns:
        Tensor: Generated image tensor.
    """
    # set eval mode
    model.eval()
    # construct sampling list for batches
    n_repeat = num_samples // num_batches
    batches_list = [num_batches] * n_repeat

    # check and convert the input labels
    if isinstance(label, int):
        label = torch.LongTensor([label] * num_samples)
    elif isinstance(label, torch.Tensor):
        label = label.type(torch.int64)
        if label.numel() == 1:
            # repeat single tensor
            # call view(-1) to avoid nested tensor like [[[1]]]
            label = label.view(-1).repeat(num_samples)
        else:
            # flatten multi tensors
            label = label.view(-1)
    elif isinstance(label, list):
        if is_list_of(label, int):
            label = torch.LongTensor(label)
            # `nargs='+'` parse single integer as list
            if label.numel() == 1:
                # repeat single tensor
                label = label.repeat(num_samples)
        else:
            raise TypeError('Only support `int` for label list elements, '
                            f'but receive {type(label[0])}')
    elif label is None:
        pass
    else:
        raise TypeError('Only support `int`, `torch.Tensor`, `list[int]` or '
                        f'None as label, but receive {type(label)}.')

    # check the length of the (converted) label
    if label is not None and label.size(0) != num_samples:
        raise ValueError('Number of elements in the label list should be ONE '
                         'or the length of `num_samples`. Requires '
                         f'{num_samples}, but receive {label.size(0)}.')

    # make label list
    label_list = []
    for n in range(n_repeat):
        if label is None:
            label_list.append(None)
        else:
            label_list.append(label[n * num_batches:(n + 1) * num_batches])

    if num_samples % num_batches > 0:
        batches_list.append(num_samples % num_batches)
        if label is None:
            label_list.append(None)
        else:
            label_list.append(label[(n + 1) * num_batches:])

    res_list = []

    # inference
    for batches, labels in zip(batches_list, label_list):
        res = model.sample_from_noise(
            None,
            num_batches=batches,
            label=labels,
            sample_model=sample_model,
            **kwargs)
        res_list.append(res.cpu())

    results = torch.cat(res_list, dim=0)

    return results


def sample_img2img_model(model, image_path, target_domain=None, **kwargs):
    """Sampling from translation models.

    Args:
        model (nn.Module): The loaded model.
        image_path (str): File path of input image.
        style (str): Target style of output image.
    Returns:
        Tensor: Translated image tensor.
    """
    assert isinstance(model, BaseTranslationModel)

    # get source domain and target domain
    if target_domain is None:
        target_domain = model._default_domain
    source_domain = model.get_other_domains(target_domain)[0]

    cfg = model._cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    # prepare data
    data = dict()
    # dirty code to deal with test data pipeline
    data['pair_path'] = image_path
    data[f'img_{source_domain}_path'] = image_path
    data[f'img_{target_domain}_path'] = image_path

    data = test_pipeline(data)
    if device.type == 'cpu':
        data = collate([data], samples_per_gpu=1)
        data['meta'] = []
    else:
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    source_image = data[f'img_{source_domain}']
    # forward the model
    with torch.no_grad():
        results = model(
            source_image,
            test_mode=True,
            target_domain=target_domain,
            **kwargs)
    output = results['target']
    return output


@torch.no_grad()
def sample_ddpm_model(model,
                      num_samples=16,
                      num_batches=4,
                      sample_model='ema',
                      same_noise=False,
                      **kwargs):
    """Sampling from ddpm models.

    Args:
        model (nn.Module): DDPM models in MMGeneration.
        num_samples (int, optional): The total number of samples.
            Defaults to 16.
        num_batches (int, optional): The number of batch size for inference.
            Defaults to 4.
        sample_model (str, optional): Which model you want to use. ['ema',
            'orig']. Defaults to 'ema'.
        noise_batch (torch.Tensor): Noise batch used as denoising starting up.
            Defaults to None.

    Returns:
        list[Tensor | dict]: Generated image tensor.
    """
    model.eval()

    n_repeat = num_samples // num_batches
    batches_list = [num_batches] * n_repeat

    if num_samples % num_batches > 0:
        batches_list.append(num_samples % num_batches)

    noise_batch = torch.randn(model.image_shape) if same_noise else None

    res_list = []
    # inference
    for idx, batches in enumerate(batches_list):
        mmcv.print_log(
            f'Start to sample batch [{idx+1} / '
            f'{len(batches_list)}]', 'mmgen')
        noise_batch_ = noise_batch[None, ...].expand(batches, -1, -1, -1) \
            if same_noise else None

        res = model.sample_from_noise(
            noise_batch_,
            num_batches=batches,
            sample_model=sample_model,
            show_pbar=True,
            **kwargs)
        if isinstance(res, dict):
            res = {k: v.cpu() for k, v in res.items()}
        elif isinstance(res, torch.Tensor):
            res = res.cpu()
        else:
            raise ValueError('Sample results should be \'dict\' or '
                             f'\'torch.Tensor\', but receive \'{type(res)}\'')
        res_list.append(res)

    # gather the res_list
    if isinstance(res_list[0], dict):
        res_dict = dict()
        for t in res_list[0].keys():
            # num_samples x 3 x H x W
            res_dict[t] = torch.cat([res[t] for res in res_list], dim=0)
        return res_dict
    else:
        return torch.cat(res_list, dim=0)
