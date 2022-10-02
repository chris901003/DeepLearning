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
