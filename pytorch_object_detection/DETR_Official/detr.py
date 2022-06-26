import colorsys
import os
import time
import numpy as np
import torch
from torch import nn
from PIL import ImageFont, ImageDraw, Image
from models import build_model
import datasets.transforms as T
from datasets.coco import make_coco_transforms
from torchvision import transforms
import json


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


class DETR(object):
    def __init__(self, args):
        # 在構建模型時會同時構建的兩個東西
        self.net = None
        self.postprocessors = None
        self.criterion = None
        # 預訓練權重位置
        self.model_path = ''
        # 類別文件位置
        self.classes_path = ''
        # 標註標界匡中內容的字體檔案位置
        self.word_type = ''
        # 在用segmentation時需要的調色板json檔案位置
        self.palette_path = ''
        # 輸入模型圖像大小
        self.input_shape = [800, 800]
        # 預測值大於這個閾值的邊界匡才會被留下
        self.confidence = 0.7
        # 運行設備
        self.cuda = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names, self.num_classes = get_classes(self.classes_path)
        # 不同類別的邊界匡會用不同顏色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        # ---------------------------------------------------#
        # lambda簡單說明
        # 在:符號前的表示要傳入的變數，用逗號隔開，分號後面的是運算式
        # 在運算式後面可以加上可遍歷的變數，然後用逗號分開
        # 在lambda前面加上map可以讓運算式逗號後面的值一一帶入lambda中，不過結果需要透過list才能拿出來，所以會在最外面加上list
        # ---------------------------------------------------#
        # colorsys是可以將RGB和YIQ/HLS/HSV颜色模式轉換的模組，當然還有更多操作
        # colorsys.hsv_to_rgb就可以可以將hsv格式轉成rgb格式，需要傳入三個值且範圍都是[0, 1]
        # colors = [[r1, g1, b1], [r2, g2, b2], ..., [rn, gn, bn]]，n = num_classes
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # 因為透過colorsys轉換到rgb後都是[0, 1]的值，所以我們乘以255變回正常的RGB
        # 這樣就可以決定每個類別要用的顏色了
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        # 構建模型
        self.generate(args)
        self.transforms = make_coco_transforms('val')

    def generate(self, args):
        # net = 預測模型
        # criterion = 計算損失用得，這裡我們不會用到
        # postprocessors = 後處理用的，將預測的結果轉換成我們需要的樣子
        self.net, self.criterion, self.postprocessors = build_model(args)
        # 加載權重，這裡只會加載模型部分的權重
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device)['models'])
        self.net = self.net.eval()
        self.net = self.net.to(self.device)

    def detect_image(self, image):
        # 在Image格式中是(寬高)，但是轉換到tensor後會變成(高寬)
        w, h = image.size
        # 保存原始圖像大小
        orig_size = torch.as_tensor([int(h), int(w)])
        # 加上batch維度同時放到設備上
        orig_size = orig_size.unsqueeze(0).to(self.device)

        # 進行圖像變換，我們只需要image的轉換就可以了
        # 提供兩種方式進行圖像轉換，第一種fps會比較穩定，但是準確度理論上來說會比較好
        images = self.transforms(image, None)[0]

        # 第二種轉換方式要依據對螢幕截圖大小決定fps，同時如果擷取大小過小可能導致預測不準確
        # images = transforms.ToTensor()(image)
        # images = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(images)

        h, w = images.shape[1], images.shape[2]
        # 保存輸入到模型中圖片大小
        target_size = torch.as_tensor([int(h), int(w)])
        # 加上batch維度同時放到設備上
        target_size = target_size.unsqueeze(0).to(self.device)
        # 加上batch_size維度
        images = images.unsqueeze(0)
        with torch.no_grad():
            images = images.to(self.device)
            # ---------------------------------------------------------
            # outputs (Dict)
            # {
            #   'pred_logits': shape [batch_size, num_queries, num_classes + 1],
            #   'pred_boxes': shape [batch_size, num_queries, 4],
            #   'pred_masks': shape [batch_size, num_queries, height, width] (在使用segmentation才會有)
            # }
            # 預測匡是相對座標
            # ---------------------------------------------------------
            outputs = self.net(images)
            results = self.postprocessors['bbox'](outputs, orig_size)
            # ---------------------------------------------------------
            # results = [
            #   {
            #       'scores': shape [num_queries],
            #       'labels': shape [num_queries],
            #       'boxes': shape [num_queries,4],
            #       'masks': shape [num_queries, 1, height, width]
            #   },
            #   {
            #       'scores': shape [num_queries],
            #       'labels': shape [num_queries],
            #       'boxes': shape [num_queries,4],
            #       'masks': shape [num_queries, 1, height, width],
            #   },
            # ]
            # results的長度就會是batch_size
            # ---------------------------------------------------------
            if 'segm' in self.postprocessors.keys():
                results = self.postprocessors['segm'](results, outputs, orig_size, target_size)
            # 根據分類類別置信度進行過濾，將置信度低於閾值的邊界匡移除
            for result in results:
                score_mask = result['scores'] > self.confidence
                result['scores'] = result['scores'][score_mask]
                result['labels'] = result['labels'][score_mask]
                result['boxes'] = result['boxes'][score_mask]
                if 'masks' in result.keys():
                    result['masks'] = result['masks'][score_mask]
            # 進行預測時只會有一張圖片，所以我們先把圖片的計算結果都拿出來，這樣就沒有batch_size的部分
            top_scores, top_labels, top_boxes = results[0]['scores'], results[0]['labels'], results[0]['boxes']
            if 'masks' in results[0].keys():
                top_masks = results[0]['masks']

            #  如果經過過濾後沒有標界匡就直接返回原圖
            if len(top_boxes) == 0:
                return image

            # 將結果都轉換成numpy格式，因為tensor格式無法進行操做
            top_conf = top_scores.cpu().numpy()
            top_label = top_labels.cpu().numpy()
            top_boxes = top_boxes.cpu().numpy()
            # top_masks shape [num_queries, height, width]
            top_masks = top_masks.squeeze(1)
            if 'masks' in results[0].keys():
                top_masks = top_masks.cpu().numpy().astype(np.uint8)

        # top_masks shape [num_queries, height, width]
        h, w = top_masks.shape[-2], top_masks.shape[-1]
        segmentation_masks = np.zeros([h, w])
        if 'masks' in results[0].keys():
            with open(self.palette_path, 'rb') as f:
                palette_dict = json.load(f)
                palette = []
                for v in palette_dict.values():
                    palette += v
            for idx, mask in enumerate(top_masks):
                label = top_label[idx]
                mask[mask == 1] = label
                mask_filter = np.where(segmentation_masks == 0)
                segmentation_masks[mask_filter] = mask[mask_filter]
            segmentation_masks = segmentation_masks.astype(np.uint8)
            segmentation_masks = Image.fromarray(segmentation_masks)
            segmentation_masks.putpalette(palette)
            segmentation_masks = segmentation_masks.convert('RGB')

        # font = 字體選擇，這裡指定字體的檔案位置
        # size = 字體大小，會根據輸入圖片做調整
        font = ImageFont.truetype(font=self.word_type, size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 邊界框寬度，也是會根據輸入圖片大小有所調整
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            # 找到對應的類別名稱
            predicted_class = self.class_names[int(c)]
            # box = (xmin, ymin, xmax, ymax)
            box = top_boxes[i]
            # score = 預測分數
            score = top_conf[i]

            # 把座標拆出來
            left, top, right, bottom = box

            # 限定範圍，不可超出圖片大小
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            # 如果想要獲取詳細框框資料以及概率分數可以從這裡岔出去
            # 下面就是把匡畫上去
            # label = string，裡面內容就是"類別名稱 分數(取小數點兩位)"
            label = '{} {:.2f}'.format(predicted_class, score)
            # 創建一個可以在給定圖像上面繪圖的實例對象，這裡會直接改image圖像
            draw = ImageDraw.Draw(image)
            # 獲取文字大小(w, h)寬高
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            # 依據高度來決定類別字體要放在哪裡
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 依據線條的寬度來決定要畫幾圈，迴圈每跑一次代表畫一圈，所以迴圈越多寬度就會越寬
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            # 類別文字的匡
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            # 添加上類別文字
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            # 刪除可繪畫實例對象
            del draw

        # 回傳畫好框框的圖片
        segmentation_image = Image.blend(segmentation_masks, image, 0.5)
        return segmentation_image
