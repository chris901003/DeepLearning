import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from models import build_model
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import decode_outputs, non_max_suppression, yolo_correct_boxes

'''
训练自己的数据集必看注释！
'''


class YOLO(object):
    # 由predict來實例化
    # 已看過
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        # 預訓練權重位置
        "model_path": 'C://Users//88698//Desktop//Pytorch//DETR_Official//weight//detr-r50-e632da11.pth',
        # .names文件位置
        "classes_path": 'model_data/coco_classes.txt',
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        # 輸入網路的大小，不是給的照片大小
        "input_shape": [800, 800],
        # ---------------------------------------------------------------------#
        #   所使用的YoloX的版本。nano、tiny、s、m、l、x
        # ---------------------------------------------------------------------#
        # 決定網路的版本，不同版本在層結構中會有不同的channel
        # 記得要對應上正確的預訓練權重，不然會無法載入
        "phi": 'nano',
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": True,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, args, **kwargs):
        # ---------------------------------------------------#
        #   初始化YOLO
        # ---------------------------------------------------#
        # 已閱讀
        # 這裡會由predict進行實例化
        # 傳入init中的args是用來構建網路用的
        # postprocessors, criterion都是在實例化模型的時候同時帶出來的，這裡先給None
        self.postprocessors = None
        self.criterion = None
        # 這裡注意一下用了python的神奇語法，讓上面定義的_defaults變成self裡面的東西，也就是下面可以用self.key拿到對應的value
        self.__dict__.update(self._defaults)
        # **kwargs沒有傳東西進來
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        # classes_path就是.names檔案的路徑
        # 回傳list以及數量
        self.class_names, self.num_classes = get_classes(self.classes_path)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        # 構建一個長度為num_classes的list裡面是tuple存有三個值，第一個小於一其他都等於1
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
        # 構建預測模型
        self.generate(args)

        # 展示設定
        show_config(**self._defaults)

    def generate(self, args, onnx=False):
        # 已看過
        # ---------------------------------------------------#
        #   生成模型
        # ---------------------------------------------------#
        # net = 後面預測的時候要用到的模型
        # criterion = 計算損失用的，這裡不會用到只是會被構建出來
        # postprocessors = 後處理用的，主要用來處理每個query最後是哪個類別以及預測分數
        self.net, self.criterion, self.postprocessors = build_model(args)
        # 設備設定
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 載入訓練權重
        self.net.load_state_dict(torch.load(self.model_path, map_location=device)['model'])
        # 設定成驗證模式
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                # 這裡如果只有單塊gpu就沒有差別
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, crop=False, count=False):
        # 已看過
        # ---------------------------------------------------#
        #   检测图片
        # ---------------------------------------------------#
        # ---------------------------------------------------#
        #   获得输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        # 轉換成RGB
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        # 通過letterbox_image可以對圖像增加灰條，可以在resize的時候讓圖片不失真
        # letterbox_image只是一個Bool變數
        # self.letterbox_image是在最前頭的defaults定義的，input_shape也是喔
        # input_size預設[640, 640]，letterbox_image預設是True
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        # preprocess_input就是做一下歸一化處理
        # transpose就是把channel維度調整到第一維度上
        # 最後加上batch_size維度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            # 將圖片轉換成tensor格式
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            # 將預測結過進行解碼
            # ---------------------------------------------------------
            # outputs (Dict)
            # {
            #   'pred_logits': shape [batch_size, num_queries, num_classes + 1],
            #   'pred_boxes': shape [batch_size, num_queries, 4]
            # }
            # 預測匡是相對座標
            # ---------------------------------------------------------

            # 將灰邊拿掉，我們需要修正預測匡位置
            # top_boxes = [xmin, ymin, xmax, ymax]且為原始圖片絕對座標
            top_boxes = yolo_correct_boxes(outputs, self.input_shape, image_shape, self.letterbox_image)
            if top_boxes[0] is None:
                return image

            # 分別把標籤、置信度以及座標位置拿出來
            device = torch.device('cuda')
            input_shapes = torch.tensor([self.input_shape]).to(device)
            # 看每個query預測的是哪個類別以及分數
            # ---------------------------------------------------------
            # results = {
            # 'scores':[batch_size, num_queries],
            # 'labels':[batch_size, num_queries],
            # 'boxes':[batch_size, num_queries,4]
            # }
            # ---------------------------------------------------------
            results = self.postprocessors['bbox'](outputs, input_shapes)
            # 轉成tensor且放到設備上，去除batch維度
            # 這裡轉到設備上是因為results是在設備上所以我們要都在同一個設備上才能做mask
            top_boxes = torch.as_tensor(top_boxes).to(device)
            top_boxes = top_boxes[0]
            # 過濾預測分數過低的query
            for result in results:
                score_mask = result['scores'] > 0.7
                result['scores'] = result['scores'][score_mask]
                result['labels'] = result['labels'][score_mask]
                top_boxes = top_boxes[score_mask].cpu().numpy()

            # 要判斷經過過濾後還有沒有目標匡
            if len(top_boxes) == 0:
                return image
            # 拿取預測類別以及預測分數
            top_label = np.array(results[0]['labels'].cpu().numpy(), dtype='int32')
            top_conf = results[0]['scores'].cpu().numpy()
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        # font = 字體選擇，這裡指定字體的檔案位置
        # size = 字體大小
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 標註匡寬度
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        # 計算一下每個類別共出現幾個，預設關閉
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        # 這裡預設也為False
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

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
            top, left, bottom, right = box

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
        return image
    
    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                  
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = decode_outputs(outputs, self.input_shape)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        #---------------------------------------------------#
        #   获得输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            
        outputs = [output.cpu().numpy() for output in outputs]
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(sub_output, [0, 2, 3, 1])[0]
            score      = np.max(sigmoid(sub_output[..., 5:]), -1) * sigmoid(sub_output[..., 4])
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200)
        print("Save to the " + heatmap_save_path)
        plt.cla()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))
        
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
