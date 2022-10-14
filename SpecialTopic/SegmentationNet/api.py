import os
import torch
import cv2
import numpy as np
import copy
import torch.nn.functional as F
from SpecialTopic.ST.utils import get_model_cfg
from SpecialTopic.ST.build import build_detector
from SpecialTopic.ST.dataset.utils import Compose


def init_module(model_type, phi, pretrained='none', num_classes=150, device='auto', with_pipeline=True,
                with_color_platte='ADE20KDataset', CLASSES=None, PALETTE=None):
    """
    Args:
        model_type: 使用的模型類型
        phi: 使用的模型大小
        pretrained: 訓練好的權重路徑，這裡不是預訓練權重
        num_classes: 分類類別數
        device: 推理設備
        with_pipeline: 是否需要直接加上資料處理流
        with_color_platte: index對應上的標註以及對應顏色
        CLASSES: 每個index對應上的類別
        PALETTE: 每個index對應上要畫的顏色
    """
    if device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(pretrained):
        pretrained = 'none'
        print('未加載訓練權重預測結果是無效的')
    model_cfg = get_model_cfg(model_type=model_type, phi=phi)
    model_cfg['pretrained'] = pretrained
    model_cfg['decode_head']['num_classes'] = num_classes
    model = build_detector(model_cfg)
    model = model.to(device)
    model = model.eval()
    if with_pipeline:
        pipeline_cfg = [
            {'type': 'MultiScaleFlipAugSegformer', 'img_scale': (2048, 512), 'flip': False,
             'transforms': [
                 {'type': 'ResizeMMlab', 'keep_ratio': True},
                 {'type': 'RandomFlipMMlab'},
                 {'type': 'NormalizeMMlab', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375],
                  'to_rgb': True},
                 {'type': 'ImageToTensorMMlab', 'keys': ['img']},
                 {'type': 'Collect', 'keys': ['img']}
             ]}
        ]
        pipeline = Compose(pipeline_cfg)
        model.pipeline = pipeline
    if with_color_platte is not None and with_color_platte != 'none' and isinstance(with_color_platte, str):
        from SpecialTopic.ST.dataset.config.segmentation_classes_platte import ADE20KDataset, FoodAndNotFood
        support_platte = {
            'ADE20KDataset': ADE20KDataset,
            'FoodAndNotFood': FoodAndNotFood
        }
        platte = support_platte.get(with_color_platte, None)
        assert platte is not None, '目前不支援該資料集'
        assert isinstance(platte, dict)
        CLASSES = platte['CLASSES']
        PALETTE = platte['PALETTE']
    model.CLASSES = CLASSES
    model.PALETTE = PALETTE
    return model


def image_draw(origin_image, seg_pred, palette, classes, opacity, with_class=False, mask=None):
    image = copy.deepcopy(origin_image)
    assert palette is not None, '需要提供調色盤，否則無法作畫'
    palette = np.array(palette)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg_pred.shape[0], seg_pred.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[(seg_pred == label) & mask[label], :] = color
    color_seg = color_seg[..., ::-1]
    image = image * (1 - opacity) + color_seg * opacity
    image = image.astype(np.uint8)
    if with_class:
        classes_set = set(seg_pred.flatten())
        color2classes = {classes[idx]: palette[idx] for idx in classes_set}
        image_height, image_width = image.shape[:2]
        ymin, ymax = image_height - 30, image_height - 10
        for idx, (class_name, color) in enumerate(color2classes.items()):
            cv2.rectangle(image, (20, ymin), (40, ymax), color[::-1].tolist(), -1)
            cv2.putText(image, class_name, (50, ymax), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (89, 214, 210), 2, cv2.LINE_AA)
            ymin -= 30
            ymax -= 30
    return image, color_seg


def detect_single_picture(model, device, image_info, threshold=0.7, opacity=0.5, CLASSES=None, PALETTE=None,
                          with_class=False, with_draw=True):
    if isinstance(image_info, str):
        image = cv2.imread(image_info)
    elif isinstance(image_info, np.ndarray):
        image = image_info
    elif isinstance(image_info, torch.Tensor):
        image = image_info.cpu().numpy()
    else:
        raise NotImplemented('目前暫不支持該圖像格式')
    origin_image = image
    data = dict(img=image)
    data = model.pipeline(data)
    image = data['img'][0].unsqueeze(dim=0)
    model, image = model.to(device), image.to(device)
    with torch.no_grad():
        output = model(image, with_loss=False)
    CLASSES = CLASSES if CLASSES is not None else model.CLASSES
    PALETTE = PALETTE if PALETTE is not None else model.PALETTE
    assert CLASSES is not None and PALETTE is not None, '需提供CLASSES與PALETTE資訊才可以做圖'
    seg_pred = F.interpolate(input=output, size=image.shape[2:], mode='bilinear', align_corners=False)
    seg_pred = F.interpolate(input=seg_pred, size=origin_image.shape[:2], mode='bilinear', align_corners=False)
    seg_pred = F.softmax(seg_pred, dim=1)
    mask = (seg_pred > threshold).squeeze(dim=0).cpu().numpy()
    seg_pred = seg_pred.argmax(dim=1)
    seg_pred = seg_pred.cpu().numpy()[0]
    if not with_draw:
        return seg_pred
    draw_image_mix, draw_image = image_draw(origin_image, seg_pred, palette=model.PALETTE, classes=model.CLASSES,
                                            opacity=opacity, with_class=with_class, mask=mask)
    return draw_image_mix, draw_image


def test():
    from PIL import Image
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pretrained = '/Users/huanghongyan/Downloads/segformer_mit-b2_512x512_160k_ade20k_20220620_114047-64e4feca.pth'
    model = init_module(model_type='Segformer', phi='m', pretrained=pretrained, num_classes=150)
    image_path = '/Users/huanghongyan/Documents/DeepLearning/mmsegmentation/data/ade/ADEChallengeData2016/imag' \
                 'es/validation/ADE_val_00000002.jpg'
    outputs = detect_single_picture(model, device, image_path)
    draw_image_mix, draw_image = outputs[0], outputs[1]
    draw_image_mix = Image.fromarray(cv2.cvtColor(draw_image_mix, cv2.COLOR_BGR2RGB))
    draw_image = Image.fromarray(cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB))
    draw_image_mix.show()
    draw_image.show()


if __name__ == '__main__':
    print('Testing segmentation api')
    test()
