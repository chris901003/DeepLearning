from SpecialTopic.ST.logger.Logger import Logger


def get_cls_from_dict(support_cls, cfg):
    cls_name = cfg.pop('type', None)
    assert cls_name is not None, 'config當中需要type資訊來指定模型'
    if cls_name not in support_cls:
        raise NotImplementedError(f'指定的 {cls_name} 模塊為實作，如果需要請自行添加')
    cls = support_cls[cls_name]
    return cls


def get_classes(classes_path):
    with open(classes_path) as f:
        class_name = f.readlines()
    class_names = [c.strip() for c in class_name]
    return class_names, len(class_names)


def get_logger(logger_name='root', logger_root='./log_save', save_info=None, send_email=False, email_sender=None,
               email_key=None):
    if send_email:
        assert email_sender is not None and email_key is not None
    logger = Logger(logger_name=logger_name, logger_root=logger_root, save_info=save_info, email_sender=email_sender,
                    email_key=email_key)
    return logger


def get_model_cfg(model_type, phi):
    from SpecialTopic.ST.model_config.SegformerConfig import SegformerConfig
    support_model = {
        'Segformer': SegformerConfig
    }
    model_cfg = support_model.get(model_type, None)
    assert model_cfg is not None, f'沒有支持{model_type}模型配置文件，如果有需要請自行添加'
    model_cfg = model_cfg.get(phi, None)
    assert model_cfg is not None, f'{model_type}沒有支持{phi}的尺寸，如果有需要請自行添加'
    return model_cfg


def to_2tuple(x):
    if isinstance(x, (int, float)):
        return x, x
    elif isinstance(x, tuple) and len(x) == 2:
        return x
    else:
        raise ValueError('無法轉換成(x, x)')


def nlc_to_nchw(x, hw_shape):
    H, W = hw_shape
    batch_size, tot_patch, channels = x.shape
    return x.transpose(1, 2).reshape(batch_size, channels, H, W)


def nchw_to_nlc(x):
    batch_size, channels, height, width = x.shape
    return x.reshape(batch_size, channels, -1).transpose(1, 2)
