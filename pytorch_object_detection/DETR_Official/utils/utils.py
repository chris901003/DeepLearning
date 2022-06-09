import numpy as np
from PIL import Image


def cvtColor(image):
    # 已看過
    # ---------------------------------------------------------#
    #   将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    # ---------------------------------------------------------#
    # 轉換RGB
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


def resize_image(image, size, letterbox_image):
    # 已看過
    # ---------------------------------------------------#
    #   对输入图像进行resize
    # ---------------------------------------------------#
    # 原始圖像大小
    iw, ih = image.size
    # 輸入網路圖像大小
    w, h = size
    if letterbox_image:
        # 增加灰邊
        scale = min(w / iw, h / ih)
        # 把圖像的最大邊長縮放到跟size一樣大
        nw = int(iw * scale)
        nh = int(ih * scale)

        # BICUBIC = 指定每個點顏色的定義方式
        image = image.resize((nw, nh), Image.BICUBIC)
        # 創造出一個有灰背景的圖
        new_image = Image.new('RGB', size, (128, 128, 128))
        # 將縮放好的圖像貼上去，不夠的地方就是灰背景
        # paste後面指定的就是從底圖的左上角座標貼上
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        # 不使用letterbox_image就是直接暴力resize
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def get_classes(classes_path):
    # ---------------------------------------------------#
    #   获得类
    # ---------------------------------------------------#
    # 已閱讀
    # 從給定的類別檔案中讀取資料，回傳list以及int數量
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def preprocess_input(image):
    # 簡單的做一下圖片預處理
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def get_lr(optimizer):
    # 已看過
    # ---------------------------------------------------#
    #   获得学习率
    # ---------------------------------------------------#
    for param_group in optimizer.param_groups:
        return param_group['lr']


def show_config(**kwargs):
    # 已閱讀
    # 就是印出一些參數
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
    