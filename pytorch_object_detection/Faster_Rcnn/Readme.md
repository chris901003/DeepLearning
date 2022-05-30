# Faster RCNN 使用方法

## 文件結構
- backbone:特徵提取網路，可以根據自己需求選擇
- network_files:Faster R-CNN網擄模型(包括Fast R-CNN和RPN模塊)
- train_utils:訓練驗證相關模塊(包括cocotools)
- mt_dataset.py:自定義數據集
- train_mobilenet.py:以MobileNetV2作為backbone進行訓練
- train_resnet50_fpn.py:以resnet50+FPN作為backbone進行訓練
- train_multi_GPU.py:針對多GPU使用
- predict.py:簡易的預測腳本，使用訓練好的權重進行預測
- pascal_voc_classes.json:pascal_voc標籤文件

## 檔案
#### train_mobilenet.py
使用mobilenet當作backbone進行訓練

#### train_res50_fpn.py
使用resnet50 FPN當作backbone進行訓練，裡面有更詳細去配置一開始的內容

#### predict.py
就是測試看看，會直接去讀取以訓練好的權重

#### draw_box_utils.py
在圖片上畫出檢測部分以及類別

#### network_files/fast_rcnn_framwork.py
- FasterRCNNBase:Faster R-CNN的基礎框架，裡面有設定backbone,rpn,roi_head
- FasterRCNN:完整帶入參數

## 註記
- objectness: 分類是前景或是背景
- pred_bbox_detail: 說明框框的min_x, min_y, max_x, max_y
- anchors: 每張圖片上的所有anchors資訊，以(0, 0)為中心，標示出min_x, min_y, max_x, max_y
- num_anchors_pre_level: 每個特徵層的anchors數量
- box_cls: 將多個特徵層的batch的bojectness拼接在一起 
  - 也就是假設batch=8，每個圖片的anchers有500個那麼原先是[8, 500, 1]會變成[4000, 1]
- box_regression: 將多個特徵層的batch的pred_bbox_detail拼接在一起
- rel_codes: 每個特徵圖上的點的邊界匡回歸參數
- boxes: 就是anchors
- concat_boxes: 將所有batch的anchors給拼接起來
- pred_boxes: anchors經過邊界匡回歸參數調整後的邊界匡位置
- proposals: 拿到pred_boxes後再把batch之間分開，也就是對多一個batch的維度
- level: 記錄分割不同特徵層上的anchors索引訊息，用不同值填充來知道是哪個特稱層的anchors
- top_n_idx: 對於objectness取出前n大的，也就是說取出最像前景的那幾個index
  - 這邊要注意index在不同特徵層上需要加上偏移，就是第一個特稱層是從0第二層是從第一層的anchors往後算
  - 最後再將不同特徵層的concat起來
- boxes_for_nms: 讓boxes可以加上偏移，讓我們再算iou時不同特徵層的不會互相干擾
- 
